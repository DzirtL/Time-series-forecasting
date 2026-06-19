import json
import matplotlib.pyplot as plt
import glob
import os

def plot_learning_curves(json_paths, save_dir="plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for path in json_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            losses = data.get("loss", [])
            
            if not losses:
                print(f"В файле {path} нет данных об ошибке.")
                continue

            filename = os.path.basename(path)
            model_name = filename.replace(".json", "")

            plt.figure(figsize=(10, 6))
            
            if "train" in filename.lower():
                plt.plot(losses, label='Train Loss', color='blue', linewidth=2)
                title = f"Кривая обучения (Train)\n{model_name}"
            elif "test" in filename.lower():
                plt.plot(losses, label='Test/Val Loss', color='orange', linewidth=2)
                title = f"Кривая валидации (Test)\n{model_name}"
            else:
                plt.plot(losses, label='Loss', color='green', linewidth=2)
                title = f"Функция потерь\n{model_name}"

            plt.title(title)
            plt.xlabel("Эпохи (итерации)")
            plt.ylabel("Loss (MSE)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            save_path = os.path.join(save_dir, f"plot_{model_name}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"График сохранен: {save_path}")

        except Exception as e:
            print(f"Ошибка при обработке {path}: {e}")

def compare_models(json_paths, save_dir="plots", metric_type="test"):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(12, 8))
    
    for path in json_paths:
        if metric_type not in path.lower():
            continue
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            losses = data.get("loss", [])
            if not losses:
                continue
            model_name = os.path.basename(path).replace(".json", "").replace("_traffic_"+metric_type, "")
            
            plt.plot(losses, label=model_name, linewidth=2)
            
        except Exception as e:
            print(f"Ошибка: {e}")

    plt.title(f"Сравнение {metric_type.capitalize()} Loss для разных архитектур")
    plt.xlabel("Эпохи")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, f"Comparison_{metric_type}_loss.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"График сравнения сохранен: {save_path}")

if __name__ == "__main__":
    runs_dir = "runs"
    json_files = glob.glob(os.path.join(runs_dir, "*.json"))
    
    if not json_files:
        print(f"JSON файлы не найдены в папке '{runs_dir}'!")
    else:
        print(f"Найдено файлов: {len(json_files)}")
        plot_learning_curves(json_files)
        compare_models(json_files, metric_type="test")
        compare_models(json_files, metric_type="train")