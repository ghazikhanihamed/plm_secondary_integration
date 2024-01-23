from xml.etree import ElementTree as ET

# Load and parse the XML file
file_path = '/mnt/data/meme.xml'

# Function to parse the XML file and extract relevant data
def parse_meme_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extracting motifs and their occurrences
    motifs = []
    for motif in root.findall(".//motif"):
        motif_id = motif.get('id')
        motif_name = motif.get('name')
        occurrences = []
        for occ in motif.findall(".//sequence"):
            seq_id = occ.get('id')
            start = occ.get('start')
            occurrences.append({'sequence_id': seq_id, 'start': start})

        motifs.append({
            'motif_id': motif_id,
            'motif_name': motif_name,
            'occurrences': occurrences
        })

    return motifs

# Parse the XML
# motifs_data = parse_meme_xml(file_path)
# motifs_data  # Display a sample of the parsed data for verification purposes



# ----- Separately

# # Path to the newly uploaded XML files
# misclassified_file_path = './meme_files/localization_common_misclassified_meme.xml'
# correctly_classified_file_path = './meme_files/localization_common_correctly_classified_meme.xml'

# # Parsing both XML files
# misclassified_motifs = parse_meme_xml(misclassified_file_path)
# correctly_classified_motifs = parse_meme_xml(correctly_classified_file_path)

# # Display a sample of the parsed data from both files for verification
# print('Misclassified motifs:')
# print(misclassified_motifs)
# print('Correctly classified motifs:')
# print(correctly_classified_motifs)



# ------- Using text output

# Reading the text files containing the MEME results for both misclassified and correctly classified sequences
misclassified_txt_path = './meme_files/solubility_common_misclassified_meme.txt'
correctly_classified_txt_path = './meme_files/solubility_common_correctly_classified_meme.txt'

# Function to extract motif information from the text file
def extract_motif_info_from_txt(file_path):
    motifs = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        motif_id = None
        for line in lines:
            if line.startswith('MOTIF'):
                motif_id = line.split()[1]
                motifs[motif_id] = {'occurrences': 0}
            elif motif_id and line.startswith(' '):
                motifs[motif_id]['occurrences'] += 1

    return motifs

# Extracting motifs from both files
misclassified_motifs_info = extract_motif_info_from_txt(misclassified_txt_path)
correctly_classified_motifs_info = extract_motif_info_from_txt(correctly_classified_txt_path)

# Display a sample of the extracted data for verification
print('Misclassified motifs:')
print(misclassified_motifs_info)
print('Correctly classified motifs:')
print(correctly_classified_motifs_info)
