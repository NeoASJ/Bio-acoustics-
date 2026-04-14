from pathlib import Path 
root = Path(r'C:\Users\HP\Downloads\Foss_hack\Data-Pool-Bio-Acoustics-\exps\cleaned\clean')
for subfolders in root.iterdir():
    count_of_files = sum( 1 for file in subfolders.iterdir() if file.is_file())
    print(f'The {subfolders.name} Species has {count_of_files} audio files')
      
