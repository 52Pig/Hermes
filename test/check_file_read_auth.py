import os

file_path = '../model_dir/model_81000.zip'

if os.path.isfile(file_path):
    try:
        with open(file_path, 'rb') as f:
            print("文件可访问且可读取。")
    except PermissionError:
        print("权限被拒绝。请检查文件权限。")
else:
    print("文件不存在。")
