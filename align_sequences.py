import os
import subprocess


def run_t_coffee_for_tasks(
    tasks, misclassified_folder, correctly_classified_folder, output_folder
):
    # Make sure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for task in tasks:
        misclassified_file = f"{task}_common_misclassified.fasta"
        correctly_classified_file = f"{task}_common_correctly_classified.fasta"

        misclassified_path = os.path.join(misclassified_folder, misclassified_file)
        correctly_classified_path = os.path.join(
            correctly_classified_folder, correctly_classified_file
        )

        if os.path.exists(misclassified_path) and os.path.exists(
            correctly_classified_path
        ):
            # Merge both files into one for alignment
            combined_file = f"{task}_combined.fasta"
            with open(combined_file, "w") as outfile:
                for fname in [misclassified_path, correctly_classified_path]:
                    with open(fname) as infile:
                        outfile.write(infile.read())
                        outfile.write("\n")

            output_path = os.path.join(output_folder, f"{task}_aligned.fasta")

            # Construct the T-Coffee command
            command = (
                f"t_coffee -in {combined_file} -outfile {output_path} -output fasta"
            )

            # Run the command
            subprocess.run(command, shell=True)

            # Remove the combined file
            os.remove(combined_file)

        else:
            print(f"Files for {task} not found. Skipping.")


# Example usage
tasks = [
    "transporters",
    "localization",
    "solubility",
    "ionchannels",
    "mp",
]

misclassified_folder = "./misclassified_sequences"
correctly_classified_folder = "./correctly_classified_sequences"
output_folder = "./aligned_sequences"

run_t_coffee_for_tasks(
    tasks, misclassified_folder, correctly_classified_folder, output_folder
)
