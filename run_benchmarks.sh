#!/bin/bash
# =============================================================================
# Thesis Benchmark Runner
# =============================================================================
# Prerequisites:
#   - Build: make clean && make
#   - Models in project root:
#       scene.gltf        (Stanford Bunny,     ~70K tris)
#       armadillo.obj      (Stanford Armadillo, ~345K tris)
#       happy_buddha.obj   (Happy Buddha,       ~1.09M tris)
#       Glykon.obj         (Glykon,             ~2.56M tris)
#   - Download Stanford models from:
#       https://graphics.stanford.edu/data/3Dscanrep/
#
# Usage:
#   chmod +x run_benchmarks.sh
#   ./run_benchmarks.sh              # run all experiments
#   ./run_benchmarks.sh 1            # run only experiment 1
#   ./run_benchmarks.sh 2            # run only experiment 2
#   ./run_benchmarks.sh 3            # run only experiment 3
#
# Output: decim_runs.csv + decim_iterations.csv (appended per run)
# Each run is repeated $REPEATS times (default 5). Median selection is manual.
# =============================================================================

set -e

BINARY=./VulkanLOD
REPEATS=${REPEATS:-5}

BUNNY="scene.gltf"
ARMADILLO="Armadillo.ply"
BUDDHA="happy_recon/happy_vrip.ply"
GLYKON="Glykon.obj"

# ALL_MODELS=("$BUNNY" "$ARMADILLO" "$BUDDHA" "$GLYKON")
# MODEL_NAMES=("bunny" "armadillo" "buddha" "glykon")

ALL_MODELS=("$BUNNY" "$ARMADILLO" "$BUDDHA")
MODEL_NAMES=("bunny" "armadillo" "buddha")


run() {
    local desc="$1"; shift
    echo ""
    echo "================================================================"
    echo "  $desc"
    echo "  CMD: $*"
    echo "================================================================"
    for r in $(seq 1 $REPEATS); do
        echo "  --- repeat $r/$REPEATS ---"
        "$@"
        echo ""
    done
}

# Check models exist
for m in "${ALL_MODELS[@]}"; do
    if [ ! -f "$m" ]; then
        echo "WARNING: Model '$m' not found. Runs using it will fail."
    fi
done

EXPERIMENT="${1:-all}"

# =============================================================================
# EXPERIMENT 1: Light Iteration Frequency
# Mode 0, ratio 0.1, sweep fullRebuildFreq = {1, 2, 3, 5, 10, 20}
# All 4 models
# =============================================================================
if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "1" ]; then
    echo ""
    echo "###################################################################"
    echo "# EXPERIMENT 1: Light Iteration Frequency Sweep"
    echo "###################################################################"

    for i in "${!ALL_MODELS[@]}"; do
        model="${ALL_MODELS[$i]}"
        name="${MODEL_NAMES[$i]}"
        for freq in 1 2 3 5 10 20; do
            run "Exp1: $name, freq=$freq" \
                env MODEL_PATH="$model" \
                    DECIM_MODE=0 \
                    DECIM_RATIO=0.1 \
                    DECIM_NUM=300 \
                    DECIM_LIGHT=$freq \
                    DECIM_LOG=1 \
                    CPU_DECIM=0 \
                    HEADLESS=1 \
                    $BINARY
        done
    done
fi

# =============================================================================
# EXPERIMENT 2: Cost Modes + CPU Baseline
# Best freq from Exp1 (default 5, override with BEST_FREQ env var)
# All 4 models x ratios {0.5, 0.1, 0.01} x modes {0, 1, 2} + CPU
# =============================================================================
if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "2" ]; then
    echo ""
    echo "###################################################################"
    echo "# EXPERIMENT 2: Cost Modes and CPU Baseline"
    echo "###################################################################"

    FREQ=${BEST_FREQ:-5}
    echo "# Using fullRebuildFreq=$FREQ (set BEST_FREQ env var to override)"

    for i in "${!ALL_MODELS[@]}"; do
        model="${ALL_MODELS[$i]}"
        name="${MODEL_NAMES[$i]}"
        for ratio in 0.5 0.1 0.01; do
            # GPU modes
            for mode in 0 1 2; do
                run "Exp2: $name, ratio=$ratio, mode=$mode (GPU)" \
                    env MODEL_PATH="$model" \
                        DECIM_MODE=$mode \
                        DECIM_RATIO=$ratio \
                        DECIM_NUM=300 \
                        DECIM_LIGHT=$FREQ \
                        DECIM_LOG=1 \
                        CPU_DECIM=0 \
                        HEADLESS=1 \
                        $BINARY
            done
            # CPU baseline (meshoptimizer)
            run "Exp2: $name, ratio=$ratio, CPU meshopt" \
                env MODEL_PATH="$model" \
                    DECIM_MODE=0 \
                    DECIM_RATIO=$ratio \
                    DECIM_NUM=0 \
                    DECIM_LOG=1 \
                    CPU_DECIM=1 \
                    HEADLESS=1 \
                    $BINARY
        done
    done
fi

# =============================================================================
# EXPERIMENT 3: Scalability
# Best mode + freq from Exp1+2 (defaults: mode 0, freq 5)
# All 4 models, ratio 0.1, GPU + CPU
# =============================================================================
if [ "$EXPERIMENT" = "all" ] || [ "$EXPERIMENT" = "3" ]; then
    echo ""
    echo "###################################################################"
    echo "# EXPERIMENT 3: Scalability"
    echo "###################################################################"

    FREQ=${BEST_FREQ:-5}
    MODE=${BEST_MODE:-0}
    echo "# Using mode=$MODE, fullRebuildFreq=$FREQ"
    echo "# (set BEST_MODE and BEST_FREQ env vars to override)"

    for i in "${!ALL_MODELS[@]}"; do
        model="${ALL_MODELS[$i]}"
        name="${MODEL_NAMES[$i]}"

        # GPU
        run "Exp3: $name, GPU (mode=$MODE, freq=$FREQ)" \
            env MODEL_PATH="$model" \
                DECIM_MODE=$MODE \
                DECIM_RATIO=0.1 \
                DECIM_NUM=300 \
                DECIM_LIGHT=$FREQ \
                DECIM_LOG=1 \
                CPU_DECIM=0 \
                HEADLESS=1 \
                $BINARY

        # CPU
        run "Exp3: $name, CPU meshopt" \
            env MODEL_PATH="$model" \
                DECIM_MODE=0 \
                DECIM_RATIO=0.1 \
                DECIM_NUM=0 \
                DECIM_LOG=1 \
                CPU_DECIM=1 \
                HEADLESS=1 \
                $BINARY
    done
fi

echo ""
echo "###################################################################"
echo "# DONE. Results appended to:"
echo "#   decim_runs.csv"
echo "#   decim_iterations.csv"
echo "###################################################################"
