import os
import hashlib
import pandas as pd

dataset_dir = os.path.join("data", "brain_tumor_mri")

train_dir = os.path.join(dataset_dir, "Training")

test_dir = os.path.join(dataset_dir, "Testing")


class_names = {
    "no tumor": 0,
    "tumor": 1
}

def collect_images(folder_path, label):
    image_records = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            
            image_records.append({
                "Path": image_path,
                "Name":filename,
                "Label": label
            })
    return image_records

def collect_dataset():
    train_records = []
    test_records = []
    
    train_records.extend(
        collect_images(os.path.join(train_dir, "no tumor"), 0)
    )
    
    train_records.extend(
        collect_images(os.path.join(train_dir, "tumor"), 1)
    )
    
    test_records.extend(
        collect_images(os.path.join(test_dir, "no tumor"), 0)
    )
    
    test_records.extend(
        collect_images(os.path.join(test_dir, "tumor"), 1)
    )
    
    return train_records, test_records


def calculate_hash(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        
    image_hash = hashlib.sha256(image_data).hexdigest()
    
    return image_hash 


def is_duplicate(image_path, seen_hashes):
    image_hash = calculate_hash(image_path)
    
    if image_hash in seen_hashes:
        return True
    
    seen_hashes.add(image_hash)
    
    return False


def check_training_duplicates(train_records):
    seen_hashes = set()
    duplicate_records = []
    
    for record in train_records:
        image_path  = record["Path"]
        
        if is_duplicate(image_path, seen_hashes):
            duplicate_records.append(record)
            
    return duplicate_records


def remove_training_duplicates(train_records):
    seen_hashes = set()
    clean_records = []

    for record in train_records:
        image_path = record["Path"]
        image_hash = calculate_hash(image_path)

        if image_hash not in seen_hashes:
            seen_hashes.add(image_hash)
            clean_records.append(record)

    return clean_records

def collect_training_hashes(train_records):
    training_hashes = set()

    for record in train_records:
        image_path = record["Path"]
        image_hash = calculate_hash(image_path)

        training_hashes.add(image_hash)

    return training_hashes

def check_test_leakage(test_records, training_hashes):
    leaked_records = []

    for record in test_records:
        image_path = record["Path"]
        image_hash = calculate_hash(image_path)

        if image_hash in training_hashes:
            leaked_records.append(record)

    return leaked_records

def check_testing_duplicates(test_records):
    seen_hashes = set()
    duplicate_records = []

    for record in test_records:
        image_path = record["Path"]
        image_hash = calculate_hash(image_path)

        if image_hash in seen_hashes:
            duplicate_records.append(record)
        else:
            seen_hashes.add(image_hash)

    return duplicate_records

def create_clean_test_records(test_records, training_hashes):
    seen_hashes = set()
    clean_test_records = []

    for record in test_records:
        image_path = record["Path"]
        image_hash = calculate_hash(image_path)

        if image_hash not in training_hashes:
            if image_hash not in seen_hashes:
                seen_hashes.add(image_hash)
                clean_test_records.append(record)

    return clean_test_records

def save_records_to_csv(records, file_path):
    dataframe = pd.DataFrame(records)

    dataframe.to_csv(file_path, index=False)
    

def main():
    train_records, test_records = collect_dataset()

    clean_train_records = remove_training_duplicates(train_records)

    training_hashes = collect_training_hashes(train_records)

    clean_test_records = create_clean_test_records(
        test_records,
        training_hashes
    )

    save_records_to_csv(
        clean_train_records,
        "data/CSVs/train_clean.csv"
    )

    save_records_to_csv(
        clean_test_records,
        "data/CSVs/test_clean.csv"
    )

    print("CSV files generated successfully!")
    print("Training images:", len(clean_train_records))
    print("Testing images:", len(clean_test_records))


if __name__ == "__main__":
    main()