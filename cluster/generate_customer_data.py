import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of rows for each strategy
num_rows = {
    # Low spending
    "data1": np.random.randint(100, 200),
    "data2": np.random.randint(150, 350),
    "data3": np.random.randint(150, 350),
    "data4": np.random.randint(500, 850),
    "data5": np.random.randint(50, 200),
    # Middel spending
    "data6": np.random.randint(800, 1500),
    "data7": np.random.randint(800, 1000),
    "data8": np.random.randint(1500, 2000),
    # High spending
    "data9": np.random.randint(300, 550),
    "data10": np.random.randint(100, 250),
    
    "data11": np.random.randint(400, 600),
}

# 1. Generate data for average spending < 100, in-store visits < 5, online purchases < 5
data1 = {
    "average_spending": np.random.uniform(1, 100, num_rows["data1"]),
    "in_store_visits": np.random.randint(1, 5, num_rows["data1"]),
    "online_purchases": np.random.randint(1, 5, num_rows["data1"]),
}

# 2. Generate data for average spending < 100, in-store visits < 2, online purchases < 8
data2 = {
    "average_spending": np.random.uniform(1, 100, num_rows["data2"]),
    "in_store_visits": np.random.randint(1, 2, num_rows["data2"]),
    "online_purchases": np.random.randint(8, 20, num_rows["data2"]),
}

# 3. Generate data for average spending < 100, in-store visits < 8, online purchases < 2
data3 = {
    "average_spending": np.random.uniform(1, 100, num_rows["data3"]),
    "in_store_visits": np.random.randint(1, 20, num_rows["data3"]),
    "online_purchases": np.random.randint(1, 2, num_rows["data3"]),
}

# 4. Generate data for average spending < 100, in-store visits >8, online purchases < 2
data4 = {
    "average_spending": np.random.uniform(1, 100, num_rows["data4"]),
    "in_store_visits": np.random.randint(8, 30, num_rows["data4"]),
    "online_purchases": np.random.randint(1, 2, num_rows["data4"]),
}

# 5. Generate data for average spending < 100, in-store visits <2, online purchases >8
data5 = {
    "average_spending": np.random.uniform(1, 100, num_rows["data5"]),
    "in_store_visits": np.random.randint(1, 2, num_rows["data5"]),
    "online_purchases": np.random.randint(8, 20, num_rows["data5"]),
}


# 6. Generate data for 100 <= average spending < 300, in-store visits < 5, online purchases < 10
data6 = {
    "average_spending": np.random.uniform(100, 300, num_rows["data6"]),
    "in_store_visits": np.random.randint(1, 5, num_rows["data6"]),
    "online_purchases": np.random.randint(1, 10, num_rows["data6"]),
}

# 7. Generate data for 100 <= average spending < 300, in-store visits < 5, online purchases < 10
data7 = {
    "average_spending": np.random.uniform(100, 300, num_rows["data7"]),
    "in_store_visits": np.random.randint(5, 30, num_rows["data7"]),
    "online_purchases": np.random.randint(0, 1, num_rows["data7"]),
}

# 8. Generate data for 100 <= average spending < 300, in-store visits < 15, online purchases < 10
data8 = {
    "average_spending": np.random.uniform(100, 300, num_rows["data8"]),
    "in_store_visits": np.random.randint(5, 30, num_rows["data8"]),
    "online_purchases": np.random.randint(5, 20, num_rows["data8"]),
}

# 9. Generate data for average spending >= 500, in-store visits < 5, online purchases < 10
data9 = {
    "average_spending": np.random.uniform(300, 450, num_rows["data9"]),
    "in_store_visits": np.random.randint(0, 5, num_rows["data9"]),
    "online_purchases": np.random.randint(0, 10, num_rows["data9"]),
}

# 10. Generate data for average spending >= 500, in-store visits > 5, online purchases < 20
data10 = {
    "average_spending": np.random.uniform(300, 450, num_rows["data10"]),
    "in_store_visits": np.random.randint(5, 30, num_rows["data10"]),
    "online_purchases": np.random.randint(10, 20, num_rows["data10"]),
}


# # 11. Low spending, frequent in-store visits, rare online purchases
# data11 = {
#     "average_spending": np.random.uniform(0, 600, num_rows["data11"]),
#     "in_store_visits": np.random.uniform(0, 30, num_rows["data11"]),
#     "online_purchases": np.random.uniform(0, 30, num_rows["data11"]),
# }


# Ensure all data arrays have consistent lengths
data1_len = len(data1["average_spending"])
data2_len = len(data2["average_spending"])
data3_len = len(data3["average_spending"])
data4_len = len(data4["average_spending"])
data5_len = len(data5["average_spending"])
data6_len = len(data6["average_spending"])
data7_len = len(data7["average_spending"])
data8_len = len(data8["average_spending"])
data9_len = len(data9["average_spending"])
data10_len = len(data10["average_spending"])
#data11_len = len(data11["average_spending"])

min_length = min(data1_len, data2_len, data3_len, data4_len, data5_len, data6_len, data7_len, data8_len,data9_len,data10_len)

# Truncate all data arrays to the minimum length and filter out rows where both in_store_visits and online_purchases are 0
all_data = pd.DataFrame({
    "average_spending": np.concatenate([
        data1["average_spending"][:min_length], data2["average_spending"][:min_length], data3["average_spending"][:min_length],
        data4["average_spending"][:min_length], data5["average_spending"][:min_length], data6["average_spending"][:min_length],
        data7["average_spending"][:min_length], data8["average_spending"][:min_length],
        data9["average_spending"][:min_length], data10["average_spending"][:min_length]
    ]),
    "in_store_visits": np.concatenate([
        data1["in_store_visits"][:min_length], data2["in_store_visits"][:min_length], data3["in_store_visits"][:min_length],
        data4["in_store_visits"][:min_length], data5["in_store_visits"][:min_length], data6["in_store_visits"][:min_length],
        data7["in_store_visits"][:min_length], data8["in_store_visits"][:min_length],data9["in_store_visits"][:min_length],
        data10["in_store_visits"][:min_length], 
    ]),
    "online_purchases": np.concatenate([
        data1["online_purchases"][:min_length], data2["online_purchases"][:min_length], data3["online_purchases"][:min_length],
        data4["online_purchases"][:min_length], data5["online_purchases"][:min_length], data6["online_purchases"][:min_length],
        data7["online_purchases"][:min_length], data8["online_purchases"][:min_length],data9["online_purchases"][:min_length],
        data10["online_purchases"][:min_length], 
    ])
})

# Filter out rows where both in_store_visits and online_purchases are 0
all_data = all_data[(all_data["in_store_visits"] > 0) | (all_data["online_purchases"] > 0)]

# Shuffle the data
all_data = all_data.sample(frac=1).reset_index(drop=True)

# Save to CSV with 2 decimal places
all_data.to_csv("sam_club_customer_data.csv", index=False, float_format="%.2f")

# Randomly sample 30 rows for markdown table
sample_data = all_data.sample(30)
markdown_table = sample_data.to_markdown(index=False, floatfmt=".2f")

# Print markdown table to standard output
print("Data generation complete. File saved as 'sam_club_customer_data.csv'.")
print("\nSample Data in Markdown Table Format:\n")
print(markdown_table)
print("Data generation complete. File saved as 'sam_club_customer_data.csv'.")
