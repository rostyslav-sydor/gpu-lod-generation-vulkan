//
// simplify.cpp -- Mesh simplification algorithm.
//
// This is an efficient implementation of the mesh simplification algorithm
// presented in the paper "Surface Simplification Using Quadric Error Metrics"
// by Michael Garland and Paul S. Heckbert.  The algorithm iteratively
// contracts pairs of vertices into a single point to remove one vertex from
// the mesh at a time.  The pair to be contracted is chosen using a quadric
// error metric that aims to minimize the error introduced by the resulting
// simplification.
//
// Sample code is provided to simplify meshes in the Wavefront .OBJ file format.
//

#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>
#include <chrono>

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>

#ifndef GLM_ENABLE_EXPERIMENTAL
#define GLM_ENABLE_EXPERIMENTAL
#endif 
#define GLM_FORCE_SWIZZLE
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include "glm/gtx/string_cast.hpp"

using glm::vec2;
using glm::vec3;
using glm::vec4;
using glm::mat4;
using std::vector;
using std::set;
using std::array;
using std::pair;
using std::max_element;
using std::sort;

using timing_clock = std::chrono::high_resolution_clock;
using duration = timing_clock::duration;

// An indexed mesh.
struct indexed_mesh
{
	vector<vec3> positions;
	vector<vec2> texCoords;
	vector<vec3> normals;
	vector<array<unsigned, 3>> faces;
};

// Statistics produced by the simplify_mesh() function.
struct simplify_mesh_statistics
{
	duration pair_find_time;
	duration vertex_update_time;
	duration pair_contract_time;
	duration face_contract_time;
};

// A N-uniform hypergraph data structure that is used to accelerate the
// contraction operation at the heart of the mesh simplification algorithm.
//
// It is a collection of vertices and hyperedges that each connect N vertices
// together. A contract() operation is provided that takes two vertex indices
// and merges them together, removing any degenerate and duplicate hyperedges
// that result.
// 
template<int N>
class contraction_graph
{
private:
	// N-way hyperedge.
	struct edge_node
	{
		// Vertex indices of the edge. Always in sorted ascending order.
		array<unsigned, N> _indices;
		// Pointer to next edge containing index i = _indices[k]
		// at the same position k, for each k.
		array<edge_node*, N> _nexts;
		// Pointer to _nexts[k] of the previous edge containing
		// index i = _indices[k], for each k.
		// First edge points to _vertex_edge_heads[i][k] for vertex i.
		array<edge_node**, N> _prevs;
		// Removed edges are not deleted from the graph, to keep edge
		// indices and pointers valid. Instead, we set this flag.
		bool _removed;

		edge_node(array<unsigned, N> const& indices)
			: _indices(indices)
			, _nexts{}
			, _prevs{}
			, _removed(false)
		{
		}
		// Return whether this edge is degenerate (two same indices).
		bool degenerate() const
		{
			for (int k = 0; k < N - 1; ++k)
				if (_indices[k] == _indices[k + 1])
					return true;
			return false;
		}
		// Replace all occurrences of index i2 with i1.
		void contract(unsigned i1, unsigned i2)
		{
			for (int k = 0; k < N; ++k) {
				auto& i = _indices[k];
				if (i == i2) i = i1;
			}
			sort(_indices.begin(), _indices.end());
		}
		unsigned index(int k) const
		{
			return _indices[k];
		}
		unsigned max_index() const
		{
			return *max_element(_indices.begin(), _indices.end());
		}
	};

	// Array of all edges. This is never resized after construction,
	// to keep indices and pointers valid.
	vector<edge_node> _edges;

	// For each vertex i, an array of N pointers to edges such that the
	// k-th pointer points to an edge with i as the k-th index.
	vector<array<edge_node*, N>> _vertex_edge_heads;

	// Link a hyperedge into the vertex-lookup structure.
	void link(edge_node* edge)
	{
		assert(edge);
		for (int k = 0; k < N; ++k) {
			auto index = edge->_indices[k];
			auto& head = _vertex_edge_heads[index][k];
			if (head != nullptr)
				head->_prevs[k] = &edge->_nexts[k];
			edge->_nexts[k] = head;
			edge->_prevs[k] = &head;
			head = edge;
		}
	}

	// Remove a hyperedge from the vertex-lookup structure.
	void unlink(edge_node* edge)
	{
		assert(edge);
		for (int k = 0; k < N; ++k) {
			auto next = edge->_nexts[k];
			auto prev = edge->_prevs[k];
			if (next != nullptr)
				next->_prevs[k] = prev;
			*prev = next;
		}
	}

	// Unlink and gather all hyperedges containing given index.
	void gather(vector<edge_node*>& edges, unsigned index)
	{
		for (int k = 0; k < N; ++k) {
			auto edge = _vertex_edge_heads[index][k];
			while (edge != nullptr) {
				unlink(edge);
				edges.push_back(edge);
				edge = _vertex_edge_heads[index][k];
			}
		}
	}

public:
	// Only a const iterator is provided, to protect edge invariants.
	struct edge_iterator
	{
	private:
		vector<edge_node> const* _edges;
		typename vector<edge_node>::const_iterator _it;

	public:
		edge_iterator(vector<edge_node> const& edges, typename vector<edge_node>::const_iterator it)
			: _edges(&edges), _it(it)
		{
			while (_it != _edges->end() && _it->_removed)
				++_it;
		}

		edge_iterator& operator++ ()
		{
			do ++_it; while (_it != _edges->end() && _it->_removed);
			return *this;
		}

		array<unsigned, N> const& operator* ()  const { return  _it->_indices; }
		array<unsigned, N> const* operator-> () const { return &_it->_indices; }
		bool operator == (edge_iterator const& that) const { return _it == that._it; }
		bool operator != (edge_iterator const& that) const { return _it != that._it; }
	};

public:
	// Construct a graph from a range of face vertex indices.
	template<class InputIterator>
	contraction_graph(InputIterator first, InputIterator last)
	{
		unsigned max_index = 0;

		// Build the edges vector.
		_edges.reserve(std::distance(first, last));
		for (auto it = first; it != last; ++it) {
			edge_node edge(*it);
			if (!edge.degenerate()) {
				_edges.push_back(edge);
				max_index = std::max(max_index, edge.max_index());
			}
		}

		// Link all edges to the per-vertex lists.
		_vertex_edge_heads.resize(max_index + 1, { nullptr });
		for (edge_node& edge : _edges) link(&edge);
	}

	size_t size() const { return _edges.size(); }

	edge_iterator begin()  const { return edge_iterator(_edges, _edges.begin()); }
	edge_iterator end()    const { return edge_iterator(_edges, _edges.end()); }
	edge_iterator cbegin() const { return edge_iterator(_edges, _edges.begin()); }
	edge_iterator cend()   const { return edge_iterator(_edges, _edges.end()); }

	bool removed(int ig) const { return _edges[ig]._removed; }

	array<unsigned, N> const& operator[] (int ig) const { return _edges[ig]._indices; }

	// Performs the contraction operation that merges two vertices (indexed by iv1 and iv2)
	// together. Hyperedges that become degenerate as a result, are removed. Any resulting
	// duplicate hyperedges are also removed. A user-provided callback is called for every
	// remaining hyperedge that was affected by the operation.
	template<class UpdateFn>
	void contract(unsigned iv1, unsigned iv2, UpdateFn update)
	{
		// TODO: Avoid vector heap allocations in the hot path.
		vector<edge_node*> edges;

		// Gather all edges containing vertex iv2.
		gather(edges, iv2);

		// Replace vertex iv2 with iv1.
		for (edge_node* edge : edges)
			edge->contract(iv1, iv2);

		// Gather all edges containing vertex iv1.
		gather(edges, iv1);

		// Sort all gathered edges by set of indices.
		// Duplicate edges end up side-by-side.
		sort(edges.begin(), edges.end(),
			[this](edge_node* e1, edge_node* e2) {
				return e1->_indices < e2->_indices;
			});

		// Mark all degenerate and duplicate edges as removed.
		array<unsigned, N> last_indices = {};
		for (edge_node* edge : edges) {
			edge->_removed = edge->degenerate() || edge->_indices == last_indices;
			last_indices = edge->_indices;
		}

		// Re-link and update all non-removed edges.
		for (edge_node* edge : edges) {
			if (!edge->_removed) {
				link(edge);
				update(static_cast<unsigned>(edge - &_edges[0]), edge->_indices);
			}
		}
	}

	// Performs the contraction operation with no update callback.
	void contract(unsigned iv1, unsigned iv2)
	{
		contract(iv1, iv2, [&](unsigned, array<unsigned, N>) {});
	}
};

// Algorithm state associated with each mesh vertex.
struct vertex_state
{
	vec3 position;
	vec2 texCoords;
	vec3 normal;
	mat4 quadric;

	vertex_state(vec3 const& position, vec2 const& texCoords, vec3 const& normal)
		: position(position)
		, texCoords(texCoords)
		, normal(normal)
		, quadric(0.0f)
	{
	}
};

// Algorithm state associated with each contractible vertex pair.
struct vertex_pair_state
{
	// Target position of the contraction.
	glm::vec4 target;

	// Quadric error introduced by the contraction.
	float cost;

	// If true, this pair state has been updated, and the cost value
	// stored in the pair selection heap is out of date.  It would be
	// expensive to find a pair from the heap, remove it, and then
	// re-insert it every time that the cost of a pair changes. Instead,
	// we mark the pair as "dirty in the heap" and fix it later if we
	// encounter it while searching the heap for the next pair to contract.
	bool heap_dirty;
};

// Create initial vertex states that track the quadric cost matrix
// associated with each vertex.
vector<vertex_state> create_vertex_states(indexed_mesh const& mesh)
{
	vector<mat4> face_quadrics;

	face_quadrics.reserve(mesh.faces.size());
	for (auto& fi : mesh.faces) {
		vec3
			a = mesh.positions[fi[0]],
			b = mesh.positions[fi[1]],
			c = mesh.positions[fi[2]];

		// Plane normal of this face.
		vec3 normal = normalize(cross(b - a, c - a));
		// Plane coefficients for this face.
		glm::vec4 plane = glm::vec4(normal, -dot(normal, a));
		// Fundamental error quadric for the plane.
		mat4 quadric = outerProduct(plane, plane);

		face_quadrics.push_back(quadric);
	}

	vector<vertex_state> vertices;
	vertices.reserve(mesh.positions.size());
	for (unsigned i = 0; i < mesh.positions.size(); ++i)
		vertices.push_back(vertex_state(mesh.positions[i], mesh.texCoords[i], vec3(0)));

	for (size_t i = 0; i < mesh.faces.size(); ++i) {
		auto& fi = mesh.faces[i];
		vertices[fi[0]].quadric += face_quadrics[i];
		vertices[fi[1]].quadric += face_quadrics[i];
		vertices[fi[2]].quadric += face_quadrics[i];
	}

	return vertices;
}
static int inv = 0;
static int half = 0;
// Update a vertex pair state from given vertex states so that it correctly
// reflects the cost of contracting that pair.
void update_pair_target_and_cost(
	vertex_pair_state& p,
	vertex_state const& v1,
	vertex_state const& v2)
{
	const glm::vec4 origin = glm::vec4(0, 0, 0, 1);

	glm::vec4 v;
	mat4 q = v1.quadric + v2.quadric;

	mat4 q2 = q;
	q2[0][3] = 0;
	q2[1][3] = 0;
	q2[2][3] = 0;
	q2[3][3] = 1;

	if (determinant(q2) > 1e-2f) {
		v = inverse(q2) * origin;
		++inv;
	}
	else {
		v = glm::vec4((v1.position + v2.position) / 2.f, 1.f);
		++half;
	}


	p.target = v;
	p.cost = dot(v, q * v);


	p.heap_dirty = true;
}

// Create vertex pair states, given vertex index pairs.
vector<vertex_pair_state> create_vertex_pair_states(
	vector<vertex_state> const& vertices,
	vector<array<unsigned, 2>> const& pairs_indices)
{
	vector<vertex_pair_state> pairs;
	pairs.reserve(pairs_indices.size());

	for (auto pair_indices : pairs_indices) {
		auto& v1 = vertices[pair_indices[0]];
		auto& v2 = vertices[pair_indices[1]];

		vertex_pair_state p;
		update_pair_target_and_cost(p, v1, v2);
		p.heap_dirty = false;
		pairs.push_back(p);
	}

	return pairs;
}

// Find initial candidate pairs for contraction, based on the given distance
// threshold value. If the threshold parameter is 0, every edge of every
// face is included.
vector<array<unsigned, 2>> find_initial_pair_indices(
	vector<vertex_state> const& vertices,
	vector<array<unsigned, 3>> const& face_indices,
	float threshold)
{
	set<array<unsigned, 2>> indices;

	for (auto& f : face_indices) {
		for (int i = 2, j = 0; j < 3; i = j++) {
			if (f[j] > f[i])
				indices.insert({ f[i], f[j] });
			else
				indices.insert({ f[j], f[i] });
		}
	}

	if (threshold > 0.f) {
		for (unsigned i = 0; i < vertices.size(); ++i) {
			for (unsigned j = i + 1; j < vertices.size(); ++j) {
				auto& u = vertices[i].position;
				auto& v = vertices[j].position;
				if (length(u - v) < threshold)
					indices.insert({ i, j });
			}
		}
	}

	return vector<array<unsigned, 2>>(indices.begin(), indices.end());
}

// Prepare the min-heap that is used to select the pairs with the lowest error.
vector<pair<float, unsigned>> create_pair_heap(
	vector<vertex_pair_state> const& pairs)
{
	vector<pair<float, unsigned>> heap;
	heap.reserve(pairs.size());
	for (unsigned i = 0; i < pairs.size(); ++i)
		heap.push_back(std::make_pair(-pairs[i].cost, i));
	make_heap(heap.begin(), heap.end());
	return heap;
}

// Find the index of the pair whose contraction introduces the smallest error.
unsigned find_lowest_cost_pair(
	vector<vertex_pair_state>& pairs,
	vector<pair<float, unsigned>>& heap,
	contraction_graph<2> const& graph)
{
	// Remove pairs from the heap until we find a suitable one.
	while (!heap.empty()) {
		pop_heap(heap.begin(), heap.end());
		auto pair_index = heap.back().second;
		heap.pop_back();

		auto& pair = pairs[pair_index];

		// Pair might have been already removed as a side effect of another
		// contraction, and the heap is out of date.  This is not a problem,
		// we can simply discard the pair and try again.
		if (graph.removed(pair_index))
			continue;

		// Pair state might have been altered as a side effect of another
		// contraction, and the heap is out of date.  Push the pair back
		// into the heap with the correct cost, and try again.
		if (pair.heap_dirty) {
			pair.heap_dirty = false;
			heap.push_back(std::make_pair(-pair.cost, pair_index));
			push_heap(heap.begin(), heap.end());
			continue;
		}

		// Found a valid pair.
		return pair_index;
	}

	return 0;
}

// Main mesh simplification procedure.
void simplify_mesh(
	indexed_mesh const& in,
	indexed_mesh& out,
	float threshold,
	int iterations,
	simplify_mesh_statistics& statistics)
{

	// Vertex states out of input geometry data.
	vector<vertex_state> vertices = create_vertex_states(in); // calc quadrics for each face and vertex 
	// Vertex indices of all initial valid pairs.
	vector<array<unsigned, 2>> pair_vertex_indices = find_initial_pair_indices(vertices, in.faces, threshold);
	// Vertex pair states out of initial valid pairs.
	vector<vertex_pair_state> pair_states = create_vertex_pair_states(vertices, pair_vertex_indices);

	// Acceleration structures for contracting pairs and faces.
	auto pair_graph = contraction_graph<2>(pair_vertex_indices.begin(), pair_vertex_indices.end());
	auto face_graph = contraction_graph<3>(in.faces.begin(), in.faces.end());

	// Acceleration structure for finding the lowest-cost pair.
	vector<pair<float, unsigned>> pair_heap = create_pair_heap(pair_states);

	duration pair_find_time = {};
	duration vertex_update_time = {};
	duration pair_contract_time = {};
	duration face_contract_time = {};

	auto t0 = timing_clock::now();
	auto t1 = t0;

	// Simplification loop. Each iteration removes one vertex.
	for (int i = 0; i < iterations; ++i) {
		// Find the lowest cost pair to contract.
		unsigned ip = find_lowest_cost_pair(pair_states, pair_heap, pair_graph);

		t1 = timing_clock::now();
		pair_find_time += t1 - t0; t0 = t1;

		// Get the vertex indices and vertex states for the chosen pair.
		auto& indices = pair_graph[ip];
		unsigned iv1 = indices[0];
		vertex_state& v1 = vertices[iv1];
		unsigned iv2 = indices[1];
		vertex_state& v2 = vertices[iv2];

		// Combine vertex v1 with v2, store result into v1.
		v1.position = { pair_states[ip].target.x, pair_states[ip].target.y,pair_states[ip].target.z };
		v1.texCoords = (v1.texCoords + v2.texCoords) / 2.0f;
		v1.normal = normalize(v1.normal + v2.normal);
		v1.quadric += v2.quadric;

		t1 = timing_clock::now();
		vertex_update_time += t1 - t0; t0 = t1;

		// Pairs: Contract vertex v2 into v1, remove degenerate and
		// duplicate pairs, and update the cost of the remaining pairs.
		pair_graph.contract(iv1, iv2,
			[&](unsigned ip, array<unsigned, 2> ivs) {
				auto& v1 = vertices[ivs[0]];
				auto& v2 = vertices[ivs[1]];
				update_pair_target_and_cost(pair_states[ip], v1, v2);
			});

		t1 = timing_clock::now();
		pair_contract_time += t1 - t0; t0 = t1;

		// Faces: Contract vertex v2 into v1, remove degenerate and
		// duplicate faces.
		face_graph.contract(iv1, iv2);

		t1 = timing_clock::now();
		face_contract_time += t1 - t0; t0 = t1;
	}

	// Extract simplified geometry.
	auto remap = vector<int>(vertices.size(), -1);

	out.positions.clear();
	out.texCoords.clear();
	out.normals.clear();
	out.faces.clear();

	for (auto const& face : face_graph) {
		array<unsigned, 3> out_face;
		for (int k = 0; k < 3; ++k) {
			auto& vertex = vertices[face[k]];
			auto& index = remap[face[k]];
			if (index < 0) {
				index = static_cast<unsigned>(out.positions.size());
				out.positions.push_back(vertex.position);
				out.texCoords.push_back(vertex.texCoords);
				out.normals.push_back(vertex.normal);
			}
			out_face[k] = (unsigned)index;
		}
		out.faces.push_back(out_face);
	}

	std::cout << "inverted: " << inv << ", halved: " << half << '\n';
	//
	statistics.pair_find_time = pair_find_time;
	statistics.vertex_update_time = vertex_update_time;
	statistics.pair_contract_time = pair_contract_time;
	statistics.face_contract_time = face_contract_time;
}

// Read indexed mesh in Wavefront .OBJ format.
bool read_mesh(std::istream& is, indexed_mesh& out)
{
	vector<vec3> positions;
	vector<vec3> normals;
	vector<array<unsigned, 6>> faces6;

	std::string line, type;

	while (std::getline(is, line)) {
		auto it = std::find_if_not(
			line.begin(), line.end(),
			[](char c) { return std::isspace(c); });

		if (it == line.end())
			continue;

		if (*it == '#')
			continue;

		std::replace(it, line.end(), '/', ' ');

		auto ss = std::istringstream(line);

		ss >> type;

		if (type == "v") {
			vec3 p;
			ss >> p.x >> p.y >> p.z;
			positions.push_back(p);
		}
		else if (type == "vn") {
			vec3 n;
			ss >> n.x >> n.y >> n.z;
			normals.push_back(n);
		}
		else if (type == "f") {
			unsigned sink;
			array<unsigned, 6> f;
			for (int i = 0; i < 6; i += 2) {
				ss >> f[i] >> sink >> f[i + 1];
				f[i]--; f[i + 1]--;
			}
			faces6.push_back(f);
		}
	}

	// Wavefront .OBJ files allow separate indexing for positions and normals
	// for each vertex of the face.  The current implementation doesn't support
	// that, so we instead convert the mesh into a uniformly indexed one by
	// averaging all the normals for the same position together.
	out.positions = std::move(positions);
	out.normals.resize(out.positions.size());
	out.faces.clear();
	for (auto& f : faces6) {
		out.normals[f[0]] += normals[f[1]];
		out.normals[f[2]] += normals[f[3]];
		out.normals[f[4]] += normals[f[5]];
		out.faces.push_back({ f[0], f[2], f[4] });
	}
	for (auto& n : out.normals) {
		n = glm::normalize(n);
	}

	return true;
}

// Write indexed mesh in Wavefront .OBJ format.
void write_mesh(std::ostream& os, indexed_mesh const& mesh)
{
	for (auto const& p : mesh.positions)
		os << "v " << p.x << " " << p.y << " " << p.z << "\n";

	for (auto const& n : mesh.normals)
		os << "vn " << n.x << " " << n.y << " " << n.z << "\n";

	for (auto const& f : mesh.faces) {
		os << "f "
			<< f[0] + 1 << "/0/" << f[0] + 1 << " "
			<< f[1] + 1 << "/0/" << f[1] + 1 << " "
			<< f[2] + 1 << "/0/" << f[2] + 1 << "\n";
	}
}

// int main(int argc, char* argv[])
// {
// 	if (argc < 3) {
// 		std::cerr << "usage: " << argv[0] << " input.obj output.obj [percent_vertices_to_keep]" << std::endl;
// 		return EXIT_FAILURE;
// 	}

// 	float ratio = 0.25f;

// 	if (argc >= 4) {
// 		auto ss = std::stringstream(argv[3]);
// 		int ratio_percent = 0;
// 		ss >> ratio_percent;
// 		ratio = std::clamp(ratio_percent, 0, 100) / 100.0f;
// 	}

// 	// Load mesh data.
// 	indexed_mesh mesh;
// 	{
// 		auto in_file = std::ifstream(argv[1]);
// 		if (!in_file.is_open()) {
// 			std::cerr << "error: unable to open input file '" << argv[1] << "'" << std::endl;
// 			return EXIT_FAILURE;
// 		}
// 		read_mesh(in_file, mesh);
// 	}

// 	// Run mesh simplification.
// 	indexed_mesh simple_mesh;
// 	simplify_mesh_statistics statistics;
// 	float threshold = 0.0f;
// 	int iterations = static_cast<int>(mesh.positions.size() * (1 - ratio));
// 	simplify_mesh(mesh, simple_mesh, threshold, iterations, statistics);

// 	// Write out simplified mesh.
// 	{
// 		auto out_file = std::ofstream(argv[2]);
// 		if (!out_file.is_open()) {
// 			std::cerr << "error: unable to open output file '" << argv[1] << "'" << std::endl;
// 			return EXIT_FAILURE;
// 		}
// 		write_mesh(out_file, simple_mesh);
// 	}

// 	// Print statistics.
// 	auto total_time
// 		= statistics.pair_find_time
// 		+ statistics.vertex_update_time
// 		+ statistics.pair_contract_time
// 		+ statistics.face_contract_time;

// 	auto to_ms = [](duration d) { return std::chrono::duration_cast<std::chrono::milliseconds>(d); };

// 	std::cout
// 	     << "\n=== mesh simplification complete ===\n"
// 	     << "\n"
// 	     << "input:\n"
// 	     << "    vertices = " << mesh.positions.size() << "\n"
// 	     << "    faces = " << mesh.faces.size() << "\n"
// 	     << "output:\n"
// 	     << "    vertices = " << simple_mesh.positions.size() << "\n"
// 	     << "    faces = " << simple_mesh.faces.size() << "\n"
// 	     << "time:\n"
// 	     << "    total = " << to_ms(total_time).count() << "\n"
// 	     << "    pair find = " << to_ms(statistics.pair_find_time).count() << "\n"
// 	     << "    vertex update = " << to_ms(statistics.vertex_update_time).count() << "\n"
// 	     << "    pair contract = " << to_ms(statistics.pair_contract_time).count() << "\n"
// 	     << "    face contract = " << to_ms(statistics.face_contract_time).count() << "\n"
// 	     ;

// 	return 0;
// }
