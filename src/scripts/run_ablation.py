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
from src.ablation.multi_head_ablation import run_progressive_multi_head_ablation
from src.config.schema import Config
from src.models.registry import create_extractor
from src.utils.model_utils import get_model_architecture_info
from src.utils.paths import get_model_short_name
from src.visualization.ablation_plots import (
    generate_all_multi_head_visualizations,
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

    # Determine which experiments to run
    run_layer_ablation_exp = phases_cfg.get("layer_screening", True)
    run_head_ablation_exp = phases_cfg.get("head_analysis", True)
    run_combination_exp = phases_cfg.get("combination", True)
    run_multi_head_exp = phases_cfg.get("multi_head", False)

    # Get target layers if specified
    target_layers: str | list[int] | None = None
    if "target_layers" in ablation_cfg and ablation_cfg.get("target_layers") is not None:
        layers = ablation_cfg.get("target_layers")
        # Handle "all" keyword
        if isinstance(layers, str) and layers.lower() == "all":
            target_layers = "all"  # Will be expanded after model loading
        # Convert to list if needed (Hydra might return ListConfig)
        elif isinstance(layers, (list, tuple)) or hasattr(layers, "__iter__"):
            target_layers = [int(layer) for layer in layers]
        else:
            target_layers = None

    # Print configuration summary
    print(f"\nModel: {config.model.model_id}")
    print(f"Tasks: {config.experiment.tasks}")
    print(f"Batch size: {config.batch_size}")
    print(f"Output: {config.output.results_root}")
    print(
        f"Experiments: Layer={run_layer_ablation_exp}, Head={run_head_ablation_exp}, "
        f"Combination={run_combination_exp}, MultiHead={run_multi_head_exp}"
    )
    if target_layers == "all":
        print("Target layers: all (will be expanded after model loading)")
    elif target_layers:
        print(f"Target layers: {target_layers}")
    else:
        print("Target layers: Not specified (will use layer ablation results or default)")

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

    # Expand "all" to full layer list
    if target_layers == "all":
        target_layers = list(range(num_layers))
        print(f"\nExpanded 'all' to {num_layers} layers: [0, 1, 2, ..., {num_layers - 1}]")

    # Results storage
    layer_results = None
    head_results = None
    combination_results = None
    multi_head_results = None
    critical_layers: list[int] | None = target_layers if isinstance(target_layers, list) else None

    # Layer-level ablation
    if run_layer_ablation_exp:
        print("\n" + "=" * 80)
        print("LAYER ABLATION: Identifying Critical Layers")
        print("=" * 80)

        layer_results = run_layer_ablation(
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
            layer_output = Path(config.output.results_root) / "ablation" / "layer"
            critical_layers_path = layer_output / "critical_layers.json"

            if critical_layers_path.exists():
                with open(critical_layers_path) as f:
                    critical_info = json.load(f)
                    critical_layers = critical_info["critical_layers"]

        # Generate layer ablation visualizations
        if config.output.save_plots and layer_results is not None:
            print("\nGenerating layer ablation visualizations...")
            layer_output = Path(config.output.results_root) / "ablation" / "layer"
            plot_layer_importance(
                layer_results,
                output_path=layer_output / "layer_importance.pdf",
            )

    # Head-level ablation
    if run_head_ablation_exp:
        print("\n" + "=" * 80)
        print("HEAD ABLATION: Analyzing Individual Attention Heads")
        print("=" * 80)

        if critical_layers is None:
            print("Warning: Critical layers not specified. Using default [14, 15, 16, 17]")
            critical_layers = [14, 15, 16, 17]

        # Get model short name for organizing results
        model_short_name = get_model_short_name(config.model.model_id)

        head_results = run_head_ablation(
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

        # Generate head ablation visualizations
        if config.output.save_plots and head_results is not None:
            print("\nGenerating head ablation visualizations...")
            head_output = Path(config.output.results_root) / "ablation" / "head" / model_short_name / "plots"
            generate_all_phase2_visualizations(
                head_results,
                output_dir=head_output,
            )

    # Head combination analysis
    if run_combination_exp:
        print("\n" + "=" * 80)
        print("HEAD COMBINATION: Analyzing Multi-Head Interactions")
        print("=" * 80)

        # Get model short name for organizing results
        model_short_name = get_model_short_name(config.model.model_id)

        combination_results = run_combination_analysis(
            model=model,
            processor=processor,
            config=config,
            alignment_heads=None,  # Load from head ablation
            tasks=config.experiment.tasks,
            output_dir=None,
            device=config.device,
            max_combinations=3,
            show_progress=True,
            model_id=model_short_name,
        )

        # Generate combination visualizations
        if config.output.save_plots and combination_results is not None:
            print("\nGenerating combination visualizations...")
            combination_output = Path(config.output.results_root) / "ablation" / "head_combination"
            plot_combination_effects(
                combination_results,
                output_path=combination_output / "combination_effects.pdf",
            )

    # Multi-head progressive ablation
    if run_multi_head_exp:
        print("\n" + "=" * 80)
        print("MULTI-HEAD PROGRESSIVE ABLATION: Testing Cooperative Effects")
        print("=" * 80)

        # Get multi-head config
        multi_head_cfg = ablation_cfg.get("multi_head", {})
        multi_head_target_layers = multi_head_cfg.get("target_layers", None)

        # Use ablation.target_layers if multi_head.target_layers not specified
        if multi_head_target_layers is None:
            multi_head_target_layers = critical_layers if critical_layers is not None else [14, 15, 16, 17]
        elif isinstance(multi_head_target_layers, str) and multi_head_target_layers.lower() == "all":
            multi_head_target_layers = list(range(num_layers))
        elif isinstance(multi_head_target_layers, (list, tuple)) or hasattr(multi_head_target_layers, "__iter__"):
            multi_head_target_layers = [int(layer) for layer in multi_head_target_layers]

        if multi_head_target_layers is None:
            print("Warning: Multi-head target layers not specified. Using default [14, 15, 16, 17]")
            multi_head_target_layers = [14, 15, 16, 17]

        # Get model short name
        model_short_name = get_model_short_name(config.model.model_id)

        # Run multi-head ablation
        multi_head_results = run_progressive_multi_head_ablation(
            model=model,
            processor=processor,
            config=config,
            target_layers=multi_head_target_layers,
            tasks=config.experiment.tasks,
            output_dir=None,
            device=config.device,
            num_heads=num_heads,
            n_heads_values=multi_head_cfg.get("n_heads_values", [1, 2, 4, 8, 16, 24, 28]),
            n_trials=multi_head_cfg.get("n_trials", 10),
            random_seed=multi_head_cfg.get("random_seed", 42),
            show_progress=True,
            model_id=model_short_name,
        )

        # Generate multi-head visualizations
        if config.output.save_plots and multi_head_results is not None:
            print("\nGenerating multi-head ablation visualizations...")
            multi_head_output = Path(config.output.results_root) / "ablation" / "multi_head" / model_short_name / "figures"
            generate_all_multi_head_visualizations(
                analysis=multi_head_results["analysis"],
                baseline=multi_head_results["results"]["baseline"],
                tasks=config.experiment.tasks,
                output_dir=multi_head_output,
                n_heads_values=multi_head_cfg.get("n_heads_values", [1, 2, 4, 8, 16, 24, 28]),
            )

    # Final summary
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT COMPLETE")
    print("=" * 80)

    results_root = Path(config.output.results_root) / "ablation"
    print(f"\nResults saved to: {results_root}")

    if layer_results is not None:
        print(f"  - Layer ablation: {results_root / 'layer'}")
    if head_results is not None:
        model_short_name = get_model_short_name(config.model.model_id)
        head_dir = results_root / "head" / model_short_name
        print(f"  - Head ablation: {head_dir}")
        print("    → summary/: Aggregated results and statistics")
        print("    → by_layer/: Results split by layer")
        print("    → by_task/: Results split by task")
        print("    → plots/: Visualization PDFs")
    if combination_results is not None:
        print(f"  - Head combination: {results_root / 'head_combination'}")
    if multi_head_results is not None:
        model_short_name = get_model_short_name(config.model.model_id)
        multi_head_dir = results_root / "multi_head" / model_short_name
        print(f"  - Multi-head ablation: {multi_head_dir}")
        print("    → figures/: Progressive effect plots, heatmaps, threshold detection")
        print("    → tables/: Summary statistics and per-task breakdown")
        print("    → raw_results.pkl: Raw experimental data")
        print("    → analysis.json: Statistical analysis results")
        print("    → summary_report.txt: Text summary report")

    if config.output.save_plots:
        print("\nKey files:")
        if layer_results is not None:
            print(f"  - Layer importance: {results_root / 'layer' / 'layer_importance.pdf'}")
        if head_results is not None:
            model_short_name = get_model_short_name(config.model.model_id)
            head_dir = results_root / "head" / model_short_name
            print(f"  - Head heatmaps: {head_dir / 'plots' / 'head_importance_*.pdf'}")
            print(f"  - Summary CSV: {head_dir / 'summary' / 'head_ablation_detailed.csv'}")
            print(f"  - Alignment heads: {head_dir / 'summary' / 'alignment_heads_summary.json'}")
        if combination_results is not None:
            print(f"  - Combination effects: {results_root / 'head_combination' / 'combination_effects.pdf'}")
        if multi_head_results is not None:
            model_short_name = get_model_short_name(config.model.model_id)
            multi_head_dir = results_root / "multi_head" / model_short_name / "figures"
            print(f"  - Progressive effect plots: {multi_head_dir / 'progressive_effect_layer_*.pdf'}")
            print(f"  - Layer-nheads heatmap: {multi_head_dir / 'layer_nheads_heatmap.pdf'}")
            print(f"  - Threshold detection: {multi_head_dir / 'threshold_detection_layer_*.pdf'}")
            print(f"  - Task-specific thresholds: {multi_head_dir / 'task_specific_thresholds_layer_*.pdf'}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
