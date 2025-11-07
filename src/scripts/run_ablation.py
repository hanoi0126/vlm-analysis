"""Main entry point for ablation experiments.

Run attention head ablation experiments on VLMs.

Usage:
    # Run all phases
    python -m src.scripts.run_ablation

    # Run specific phase with Hydra overrides
    python -m src.scripts.run_ablation ablation.phases.layer_screening=false ablation.phases.combination=false

    # Override model
    python -m src.scripts.run_ablation model.model_id=Qwen/Qwen2.5-VL-7B-Instruct

    # Override tasks
    python -m src.scripts.run_ablation experiment.tasks=[color,angle,shape]
"""

import json
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ablation import run_combination_analysis, run_head_ablation, run_layer_ablation
from src.config.schema import Config
from src.models.registry import create_extractor
from src.utils.model_utils import get_model_architecture_info
from src.utils.paths import get_model_short_name
from src.visualization.ablation_plots import (
    generate_all_phase2_visualizations,
    plot_combination_effects,
    plot_layer_importance,
)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run attention head ablation experiments.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 80)
    print("Attention Head Ablation Experiments")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Convert to Pydantic model for validation
    config = Config(**OmegaConf.to_container(cfg, resolve=True))  # type: ignore[arg-type]

    # Get ablation config
    ablation_cfg = cfg.get("ablation", {})
    phases_cfg = ablation_cfg.get("phases", {})

    # Determine which phases to run
    run_phase1 = phases_cfg.get("layer_screening", True)
    run_phase2 = phases_cfg.get("head_analysis", True)
    run_phase3 = phases_cfg.get("combination", True)

    # Get target layers if specified
    target_layers = None
    if "target_layers" in ablation_cfg and ablation_cfg.get("target_layers") is not None:
        layers = ablation_cfg.get("target_layers")
        # Convert to list if needed (Hydra might return ListConfig)
        if isinstance(layers, (list, tuple)) or hasattr(layers, "__iter__"):
            target_layers = list(layers)
        else:
            target_layers = None

    # Print configuration summary
    print(f"\nModel: {config.model.model_id}")
    print(f"Tasks: {config.experiment.tasks}")
    print(f"Batch size: {config.batch_size}")
    print(f"Output: {config.output.results_root}")
    print(f"Phases: Phase1={run_phase1}, Phase2={run_phase2}, Phase3={run_phase3}")
    if target_layers:
        print(f"Target layers: {target_layers}")
    else:
        print("Target layers: Not specified (will use Phase 1 results or default)")

    # Load model
    print("\n" + "=" * 80)
    print("Loading model...")
    print("=" * 80)

    extractor = create_extractor(config.model)
    model = extractor.model
    processor = extractor.processor

    print(f"Model loaded: {config.model.model_id}")
    try:
        device_info = next(model.parameters()).device  # type: ignore[union-attr]
        print(f"Device: {device_info}")
    except StopIteration:
        print(f"Device: {config.device}")

    # Detect model architecture
    arch_info = get_model_architecture_info(model)  # type: ignore[arg-type]
    num_layers = arch_info["num_layers"]
    num_heads = arch_info["num_heads"]
    print("\nModel architecture:")
    print(f"  Number of layers: {num_layers}")
    print(f"  Number of attention heads (Q): {num_heads}")

    # Results storage
    phase1_results = None
    phase2_results = None
    phase3_results = None
    critical_layers = target_layers

    # Phase 1: Layer-level ablation
    if run_phase1:
        print("\n" + "=" * 80)
        print("PHASE 1: Layer-Level Ablation")
        print("=" * 80)

        phase1_results = run_layer_ablation(
            model=model,
            processor=processor,
            config=config,
            tasks=config.experiment.tasks,
            output_dir=None,  # Use default from config
            device=config.device,
            num_layers=num_layers,  # Auto-detected from model
            show_progress=True,
        )

        # Load critical layers if not specified
        if critical_layers is None:
            phase1_output = Path(config.output.results_root) / "ablation" / "phase1"
            critical_layers_path = phase1_output / "critical_layers.json"

            if critical_layers_path.exists():
                with open(critical_layers_path) as f:
                    critical_info = json.load(f)
                    critical_layers = critical_info["critical_layers"]

        # Generate Phase 1 visualizations
        if config.output.save_plots and phase1_results is not None:
            print("\nGenerating Phase 1 visualizations...")
            phase1_output = Path(config.output.results_root) / "ablation" / "phase1"
            plot_layer_importance(
                phase1_results,
                output_path=phase1_output / "layer_importance.pdf",
            )

    # Phase 2: Head-level ablation (PRIORITY)
    if run_phase2:
        print("\n" + "=" * 80)
        print("PHASE 2: Head-Level Ablation (PRIORITY)")
        print("=" * 80)

        if critical_layers is None:
            print("Warning: Critical layers not specified. Using default [14, 15, 16, 17]")
            critical_layers = [14, 15, 16, 17]

        # Get model short name for organizing results
        model_short_name = get_model_short_name(config.model.model_id)

        phase2_results = run_head_ablation(
            model=model,
            processor=processor,
            config=config,
            target_layers=critical_layers,
            tasks=config.experiment.tasks,
            output_dir=None,
            device=config.device,
            num_heads=num_heads,  # Auto-detected from model
            show_progress=True,
            n_bootstrap=ablation_cfg.get("statistical", {}).get("bootstrap_samples", 1000),
            n_permutations=1000,
            model_id=model_short_name,
        )

        # Generate Phase 2 visualizations
        if config.output.save_plots and phase2_results is not None:
            print("\nGenerating Phase 2 visualizations...")
            phase2_output = Path(config.output.results_root) / "ablation" / "phase2" / model_short_name / "plots"
            generate_all_phase2_visualizations(
                phase2_results,
                output_dir=phase2_output,
            )

    # Phase 3: Combination analysis
    if run_phase3:
        print("\n" + "=" * 80)
        print("PHASE 3: Multi-Head Combination Analysis")
        print("=" * 80)

        phase3_results = run_combination_analysis(
            model=model,
            processor=processor,
            config=config,
            alignment_heads=None,  # Load from Phase 2
            tasks=config.experiment.tasks,
            output_dir=None,
            device=config.device,
            max_combinations=3,
            show_progress=True,
        )

        # Generate Phase 3 visualizations
        if config.output.save_plots and phase3_results is not None:
            print("\nGenerating Phase 3 visualizations...")
            phase3_output = Path(config.output.results_root) / "ablation" / "phase3"
            plot_combination_effects(
                phase3_results,
                output_path=phase3_output / "combination_effects.pdf",
            )

    # Final summary
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT COMPLETE")
    print("=" * 80)

    results_root = Path(config.output.results_root) / "ablation"
    print(f"\nResults saved to: {results_root}")

    if phase1_results is not None:
        print(f"  - Phase 1: {results_root / 'phase1'}")
    if phase2_results is not None:
        model_short_name = get_model_short_name(config.model.model_id)
        phase2_dir = results_root / "phase2" / model_short_name
        print(f"  - Phase 2: {phase2_dir}")
        print("    → summary/: Aggregated results and statistics")
        print("    → by_layer/: Results split by layer")
        print("    → by_task/: Results split by task")
        print("    → plots/: Visualization PDFs")
    if phase3_results is not None:
        print(f"  - Phase 3: {results_root / 'phase3'}")

    if config.output.save_plots:
        print("\nKey files:")
        if phase1_results is not None:
            print(f"  - Layer importance: {results_root / 'phase1' / 'layer_importance.pdf'}")
        if phase2_results is not None:
            model_short_name = get_model_short_name(config.model.model_id)
            phase2_dir = results_root / "phase2" / model_short_name
            print(f"  - Head heatmaps: {phase2_dir / 'plots' / 'head_importance_*.pdf'}")
            print(f"  - Summary CSV: {phase2_dir / 'summary' / 'head_ablation_detailed.csv'}")
            print(f"  - Alignment heads: {phase2_dir / 'summary' / 'alignment_heads_summary.json'}")
        if phase3_results is not None:
            print(f"  - Combination effects: {results_root / 'phase3' / 'combination_effects.pdf'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
