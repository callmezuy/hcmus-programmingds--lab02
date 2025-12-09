# NumPy-Based Amazon Recommendation System

## 1\. Tổng quan dự án

Dự án này xây dựng một hệ thống gợi ý sản phẩm (Recommendation System) cho ngành hàng **Beauty Products** trên Amazon. Điểm đặc biệt của dự án là việc **implement toàn bộ quy trình từ con số 0 chỉ sử dụng thư viện NumPy**.

Dự án không sử dụng Pandas hay các Framework Deep Learning, nhằm mục tiêu tối ưu hóa khả năng tư duy đại số tuyến tính, kỹ thuật vectorization và xử lý ma trận thưa (Sparse Matrix) để giải quyết bài toán gợi ý sản phẩm thực tế.

## 2\. Mục lục

1.  Project Overview
2.  Table of Contents
3.  Introduction
4.  Dataset Overview
5.  Method (Quy trình & Thuật toán) *(To be updated)*
6.  Installation & Setup *(To be updated)*
7.  Usage *(To be updated)*
8.  Results *(To be updated)*
9.  Project Structure *(To be updated)*
10. Challenges & Solutions *(To be updated)*
11. Future Improvements *(To be updated)*
12. Contributors *(To be updated)*
13. License *(To be updated)*

## 3\. Giới thiệu

### 3.1. Mô tả bài toán

Trong thương mại điện tử, người dùng thường bị choáng ngợp bởi số lượng sản phẩm khổng lồ. Bài toán đặt ra là: *"Làm thế nào để dự đoán mức độ yêu thích (Rating) của một người dùng cụ thể đối với một sản phẩm mà họ chưa từng tương tác?"*.
Dự án này giải quyết vấn đề trên bằng cách xây dựng mô hình **Collaborative Filtering (Lọc cộng tác)** dựa trên lịch sử đánh giá của người dùng.

### 3.2. Động lực & Ứng dụng

  * **Thực tế:** Hệ thống gợi ý giúp tăng doanh thu cho Amazon bằng cách cá nhân hóa trải nghiệm mua sắm (Cross-selling/Up-selling).
  * **Học thuật:** Đây là cơ hội tuyệt vời để thực hành xử lý dữ liệu lớn và tính toán ma trận hiệu năng cao (High-performance matrix computation) mà không phụ thuộc vào các thư viện có sẵn như Pandas hay Scikit-surprise.

### 3.3. Mục tiêu cụ thể

  * Xây dựng pipeline xử lý dữ liệu thô (CSV) và chuyển đổi sang dạng ma trận User-Item bằng `numpy.genfromtxt` và Fancy Indexing.
  * Cài đặt thuật toán đo độ tương đồng (Cosine Similarity) bằng các phép toán đại số tuyến tính thuần túy.
  * Dự đoán Rating và đưa ra danh sách Top-K sản phẩm gợi ý cho người dùng.
  * Trực quan hóa phân phối Rating và độ thưa của dữ liệu bằng Matplotlib/Seaborn.

## 4\. Dataset

### 4.1. Nguồn dữ liệu

  * **Dataset Name:** Amazon Product Data - Beauty Category.
  * **Nguồn:** Amazon Reviews Data (được cung cấp trong phạm vi bài tập Homework 2).

### 4.2. Mô tả đặc trưng

Dữ liệu bao gồm các thông tin tương tác giữa người dùng và sản phẩm:

  * **User ID:** Mã định danh duy nhất của khách hàng (Dạng chuỗi ký tự - String).
  * **Product ID (ASIN):** Mã định danh duy nhất của sản phẩm (Dạng chuỗi ký tự - String).
  * **Rating:** Điểm đánh giá của khách hàng (Thang điểm 1.0 - 5.0).
  * **Timestamp:** Thời gian thực hiện đánh giá (Unix Timestamp).

### 4.3. Đặc điểm dữ liệu

  * **Sparsity (Độ thưa):** Ma trận tương tác rất thưa (Sparse), vì một người dùng chỉ đánh giá một phần rất nhỏ trong tổng số hàng triệu sản phẩm. Đây là thách thức chính khi lưu trữ và tính toán bằng NumPy array thông thường.
  * **Format:** Dữ liệu đầu vào là file CSV/JSON, yêu cầu kỹ thuật tiền xử lý để chuyển đổi (Encoding) từ chuỗi (ID) sang chỉ số (Index) số học để tính toán.
