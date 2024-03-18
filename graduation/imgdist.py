import os
import shutil

def organize_files(source_folder, low_folder, high_folder):
    # 폴더가 없다면 생성
    if not os.path.exists(low_folder):
        os.makedirs(low_folder)
    if not os.path.exists(high_folder):
        os.makedirs(high_folder)

    # 소스 폴더에서 모든 파일 목록 가져오기
    files = os.listdir(source_folder)

    # 파일을 LR 또는 HR 폴더로 이동
    for file in files:
        if 'SRF_4_LR' in file:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(low_folder, file)
            shutil.move(source_path, destination_path)
            print(f"Moved {file} to 'Low' folder.")
        elif 'SRF_4_HR' in file:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(high_folder, file)
            shutil.move(source_path, destination_path)
            print(f"Moved {file} to 'High' folder.")

# 소스 폴더, 'Low' 폴더, 'High' 폴더 경로 설정
source_folder_path = "/home/ksp/Desktop/Diff/data/test/Set5/image_SRF_4/"
low_folder_path = "/home/ksp/Desktop/Diff/data/test/Set5/LRbicx4/"
high_folder_path = "/home/ksp/Desktop/Diff/data/test/Set5/original/"

# 파일 정리 실행
organize_files(source_folder_path, low_folder_path, high_folder_path)