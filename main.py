import pandas as pd
import re
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Excel dosyasından fiyat bilgilerini çekip bir listeye ekliyorum
df = pd.read_excel('listings.xlsx')
unfiltered_prices = df['Price'].tolist()

# Fiyatlar içerisinden gereksiz kelime harf ve noktalamayı kaldırıyorum
# Aynı zamanda gerçekçi olmayan fiyat bilgilerini daha iyi bir sonuç için çıkarıyorum
prices = []
for price in unfiltered_prices:
    price_str = re.sub(r'[^0-9]', '', price)
    integer_price = int(price_str)
    if 1000 <= integer_price <= 100000:
        prices.append(integer_price)
price_array = np.array(prices).reshape(-1, 1)

# K-Kümeler algoritması ile 3 kümeye bölüyorum
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(price_array)
clusters = kmeans.labels_

# K-Küemelr algoritmasından sonra fiyat bilgilerinin ilanlara dağılımını görmek için bir grafik çıkartıyorum
plt.figure(figsize=(10, 6))
plt.scatter(range(len(price_array)), price_array.flatten(), c=clusters, cmap='viridis', label='Price Cluster')
plt.title('Scatter Plot of Prices by KMeans Clusters')
plt.xlabel('Index')
plt.ylabel('Price (TRY)')
plt.show()

# Kümeleri "Ucuz", "Orta" ve "Pahalı" olacak şekilde isimlendiriyorum
price_categories = {0: 'medium', 1: 'cheap', 2: 'expensive'}
categorized_prices = [(price, price_categories[cluster]) for price, cluster in zip(prices, clusters)]

cheap_prices = np.array([price for price, category in categorized_prices if category == 'cheap'])
medium_prices = np.array([price for price, category in categorized_prices if category == 'medium'])
expensive_prices = np.array([price for price, category in categorized_prices if category == 'expensive'])

# "Ucuz", "Orta" ve "Pahalı" kümeleri grafikde gösteriyorum
plt.figure(figsize=(10, 6))
plt.scatter(range(len(cheap_prices)), cheap_prices, color='blue', label='Cheap')
plt.scatter(range(len(medium_prices)), medium_prices, color='orange', label='Medium')
plt.scatter(range(len(expensive_prices)), expensive_prices, color='red', label='Expensive')

plt.title('Price Clusters: Cheap, Medium, and Expensive')
plt.xlabel('Index')
plt.ylabel('Price (TRY)')
plt.legend()
plt.grid(True)
plt.show()


# "Ucuz", "Orta" ve "Pahalı" kümelerin min, max, ortalama ve standart sapmasını bulup yazdıran fonksiyon
def calculate_statistics(prices, category_name):
    if prices.size > 0:
        print(f"{category_name} Prices - Lowest: {prices.min()}, Highest: {prices.max()}, "
              f"Average: {prices.mean():.2f}, Standard Deviation: {prices.std():.2f}")
    else:
        print(f"No prices available for {category_name} category.")

calculate_statistics(cheap_prices, "Cheap")
calculate_statistics(medium_prices, "Medium")
calculate_statistics(expensive_prices, "Expensive")


# Her bir kümedeki toplam fiyat sayısını bulan kod grubu
df = pd.DataFrame(categorized_prices, columns=["Price (TRY)", "Category"])
category_counts = df["Category"].value_counts()
print("Count of each category:")
print(category_counts)

# Sonuçları Excel'e gönderen fonksiyon
df.to_excel("kmeans_price_list.xlsx", index=False)
