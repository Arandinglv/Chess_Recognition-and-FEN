"""
批量rename
原始图像存放在Origin下
Rename后的放在Rename下
"""

import os
import shutil


dataset_path = "../dataset/Origin"
new_dataset_path = "../dataset/Rename"

if not os.path.exists(new_dataset_path):
    os.makedirs(new_dataset_path)

folder_mapping = {
    '00rook black': '00rook black',
    '10king white': '10king white',
    '11sodier white': '11sodier white',
    '12empty': '12empty',
    '01knight black': '01knight black',
    '02bishop black': '02bishop black',
    '03queen black': '03queen black',
    '04king black': '04king black',
    '05sodier black': '05sodier black',
    '06rook white': '06rook white',
    '07knight white': '07knight white',
    '08bishop white': '08bishop white',
    '09queen white': '09queen white'
}

for old_folder_name, new_folder_name in folder_mapping.items():
    old_folder_path = os.path.join(dataset_path, old_folder_name)
    new_folder_path = os.path.join(new_dataset_path, new_folder_name)

    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    if os.path.isdir(old_folder_path):
        files = os.listdir(old_folder_path)

        for idx, filename in enumerate(files):
            new_name = f"{idx + 1:04d}.jpg"
            src = os.path.join(old_folder_path, filename)
            dst = os.path.join(new_folder_path, new_name)
            shutil.copy(src, dst)
            print(f"Copy and rename from {src} to {dst}")

print("Successful")



