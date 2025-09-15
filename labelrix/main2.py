import json
import segment_lines
import annotation_filters
import score
import os


def list_files(dir_path):
    """Return a list of file names (not directories) in dir_path."""
    return [f.split("_")[1] + ".json" for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]


files = list_files("/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/Llama-3.3-70B-Instruct-per_page_votes")
print(files)

def main():
    MODEL = "meta-llama/Llama-3.3-70B-Instruct"
    MODEL_DIR = "Llama-3.3-70B-Instruct"
    benchmark_file_list_path = "/Volumes/MyDataDrive/thesis/code-2/src/benchmark-file-list.json"
    original_tar_json_file_dir = "/Volumes/MyDataDrive/thesis/code-2/testset-documents-ocr-idp/"
    out_path = "/Volumes/MyDataDrive/thesis/code-2/src/weak-labels-algo/"

    with open(benchmark_file_list_path, "r") as benchmark_file_list_file:
        data = json.load(benchmark_file_list_file)
    
    votes_out_dir = out_path + MODEL_DIR + "-per_page_votes"
    votes_merged_dir = out_path + MODEL_DIR + "-per_page_votes_merged"

    for key, value in data.items():
        filename = key
        pages = value['pages']
        file_path = original_tar_json_file_dir + value['file_path']
        file_value = value['file_path'].split("/")[-1]

        if file_value not in files:
            # Extract votes using LLMs
            segment_lines.extract_votes(file_path,pages,MODEL, out_dir=votes_out_dir)
    
    # Here we apply filtering
    annotation_filters.process_directory_to_json(
        in_dir=votes_out_dir,
        out_dir=votes_merged_dir,
        iou_thresh=0.5,
        overlap_thresh=0.9
    )

    # score_dir = out_path + MODEL_DIR + "/"
    # score.score_all_jsons_global(votes_merged_dir, out_dir=score_dir)


if __name__ == "__main__":
    main()