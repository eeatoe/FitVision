import os
import json

def merge_json_files(input_dir, output_file):
    """
    Объединяет все JSON-файлы из указанной директории в один JSON-файл.

    :параметр input_dir: Путь к директории, содержащей JSON-файлы.
    :параметр output_file: Путь к результирующему JSON-файлу.
    """
    merged_data = []

    # Обход всех файлов в директории
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)

                # Чтение JSON-файла
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Добавляем содержимое файла без дополнительных данных
                        merged_data.extend(data)  # Распаковываем содержимое
                except Exception as e:
                    print(f"Ошибка чтения файла {file_path}: {e}")

    # Сохранение объединённых данных в выходной файл
    try:
        with open(output_file, 'w') as f:
            json.dump(merged_data, f, indent=4)
        print(f"Объединенные данные сохранены в файл: {output_file}")
    except Exception as e:
        print(f"Ошибка записи файла {output_file}: {e}")
