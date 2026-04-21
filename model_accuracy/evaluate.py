import os
import yaml
from pathlib import Path
from ultralytics import YOLO

def main():
    # Setup paths
    base_dir = Path(__file__).parent.resolve()
    dataset_dir = base_dir / "dataset"
    yaml_path = dataset_dir / "data.yaml"
    custom_yaml_path = base_dir / "eval_data.yaml"
    model_path = base_dir.parent / "best.pt"
    
    # Ensure dataset exists
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found at {dataset_dir}")
        return
        
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # Read original yaml and fix paths
    print("\nPre-processing dataset configuration...")
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # We explicitly set the `path` directive to the absolute path of the dataset folder.
        # This makes Yolo look for 'train', 'val', 'test' inside this path.
        data['path'] = str(dataset_dir)
        data['train'] = 'train/images'
        data['val'] = 'valid/images'
        
        # Test might not exist but let's correct it as well
        if 'test' in data:
            data['test'] = 'test/images'
        
        # Write to a new yaml file
        with open(custom_yaml_path, 'w') as f:
            yaml.dump(data, f)
        print(f"Created updated dataset configuration at {custom_yaml_path}")
        
    except Exception as e:
        print(f"Error parsing dataset '{yaml_path}': {e}")
        return

    # Load and Evaluate Model
    print(f"\nLoading model from {model_path}...")
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'ultralytics' is installed. Run: pip install ultralytics")
        return

    print("\n--- Running Validation on the valid set ---")
    # Evaluate the model on the validation set specified in yaml under `val` 
    metrics = model.val(data=str(custom_yaml_path))

    # The accuracy metrics
    print("\n" + "="*40)
    print("=== MODEL ACCURACY METRICS ON VALIDATION SET ===")
    print("="*40)
    print(f"Mean Average Precision (mAP) @ 0.5-0.95: {metrics.box.map:.4f}")
    print(f"Mean Average Precision (mAP) @ 0.5:      {metrics.box.map50:.4f}")
    print(f"Mean Average Precision (mAP) @ 0.75:     {metrics.box.map75:.4f}")
    
    # For one class, or multi-class, let's just show the global mean using the numpy arrays
    print(f"Precision (mean):                        {metrics.box.p.mean():.4f}")
    print(f"Recall (mean):                           {metrics.box.r.mean():.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
