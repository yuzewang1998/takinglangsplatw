import shutil
import os
import subprocess
import os
from os.path import exists
from pathlib import Path
def backup_codebase(outputdir : Path) -> None:
    # read the file list need to copy, ignore file is in .gitignore
    backup_code_output_path = os.path.join(outputdir,'code_backup')
    os.makedirs(backup_code_output_path,exist_ok=True)
    result = subprocess.run(['git', 'ls-files'], stdout=subprocess.PIPE, text=True)
    tracked_files = result.stdout.splitlines()
    tracked_files.remove('segment-anything-langsplat')
    # print(tracked_files)
    for file_path in tracked_files:
        # Create any missing directories in the destination
        destination_path = os.path.join(backup_code_output_path, file_path)
        destination_dir = os.path.dirname(destination_path)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        shutil.copy(file_path, destination_path)



if __name__ == "__main__":

    output_path = Path("/home/wangyz/Documents/projects/0working/langsplat-w/output/test")
    backup_codebase(output_path)

