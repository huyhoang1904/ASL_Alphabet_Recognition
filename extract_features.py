import pandas as pd
import shutil
import os

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN FILE
# ==========================================
ORIGINAL_CSV = "train_data_asl (1).csv"   # File gốc chứa dữ liệu cũ
CLONE_CSV = "train_data_asl_clone.csv"    # File clone sẽ ghi đè dữ liệu mới
NEW_DATA_CSV = "new_data.csv"             # File tọa độ T, M, N, K mới trích xuất

# ==========================================
# 0. TẠO FILE CLONE ĐỂ BẢO VỆ DỮ LIỆU GỐC
# ==========================================
print("0. Đang tạo bản sao (clone) của file dữ liệu gốc...")
if os.path.exists(ORIGINAL_CSV):
    shutil.copy2(ORIGINAL_CSV, CLONE_CSV)
    print(f" -> Đã clone thành công sang file: {CLONE_CSV}\n")
else:
    print(f"[LỖI] Không tìm thấy file gốc '{ORIGINAL_CSV}'!")
    exit()

# ==========================================
# 1. ĐỌC DỮ LIỆU TỪ FILE CLONE VÀ FILE MỚI
# ==========================================
print("1. Đang tải dữ liệu lên bộ nhớ...")

# Đọc file cũ: Ép hiểu là không có tiêu đề (header=None)
# VÀ loại bỏ luôn dòng 0 chứa các số '0, 1, 2...' (skiprows=1)
df_old = pd.read_csv(CLONE_CSV, header=None, skiprows=1)

# Đọc file mới: Bản chất file này đã tinh khiết không có tiêu đề
df_new = pd.read_csv(NEW_DATA_CSV, header=None)

# ==========================================
# 2. XỬ LÝ LỌC VÀ GỘP TOÀN BỘ DỮ LIỆU
# ==========================================
print("2. Đang cắt bỏ dữ liệu cũ và gộp toàn bộ dữ liệu mới...")
classes_to_replace = ['T', 'M', 'N', 'K', 'S']

# Giữ lại các nhãn KHÔNG thuộc T, M, N, K từ file cũ (cột 0 là nhãn chữ cái)
df_kept = df_old[~df_old[0].isin(classes_to_replace)]

# Lọc chắc chắn chỉ lấy T, M, N, K từ file mới (đề phòng file lẫn dữ liệu rác)
df_new_filtered = df_new[df_new[0].isin(classes_to_replace)]

# Gộp TOÀN BỘ dữ liệu mới vào dữ liệu cũ (không giới hạn số lượng)
df_final = pd.concat([df_kept, df_new_filtered], ignore_index=True)

# ==========================================
# 3. XÁO TRỘN VÀ GHI ĐÈ LẠI VÀO FILE CLONE
# ==========================================
print("3. Đang xáo trộn (shuffle) và ghi xuất file...")
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

# Ghi đè vào file clone với định dạng thuần khiết (không index, không header)
df_final.to_csv(CLONE_CSV, index=False, header=False)

print("-" * 50)
print("ĐÃ HỢP NHẤT DỮ LIỆU THÀNH CÔNG!")
print(f"File gốc '{ORIGINAL_CSV}' được giữ nguyên, an toàn tuyệt đối.")
print(f"Dữ liệu hoàn chỉnh đã lưu tại '{CLONE_CSV}'.")
print(f"Tổng số mẫu sau khi gộp: {len(df_final)}")
print(f"\nKiểm tra lại số lượng các class vừa thay máu:")
print(df_final[df_final[0].isin(classes_to_replace)][0].value_counts())
print("-" * 50)