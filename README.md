# E-Ticaret Müşteri Churn Analizi

## Model Doğrulama ve İyileştirme Süreci

Bu bölümde, churn tahmin modelinin güvenilirliğini değerlendirmek, olası veri sızıntılarını tespit etmek ve modelin kararlarını açıklamak için izlenecek adımlar ele alınmaktadır.

### 1. Veri Güvenilirliği ve Data Leakage Kontrolü

Modelleme öncesinde veri setindeki tekrarlanan kayıtlar ve hedef değişkenle ilişkili olası veri sızıntıları incelenmelidir. Önceki analizde, `CustomerID` sütunu hariç tutulduğunda 557 tekrarlanan kayıt tespit edilmiştir. Ancak aynı özelliklere sahip kayıtların bulunması, tek başına veri hatası olduğu anlamına gelmez.

#### 1.1. Tekrarlanan Kayıtların İncelenmesi

Aşağıdaki kod, aynı özelliklere ve aynı churn etiketine sahip kayıtları, aynı özelliklere sahip tüm kayıtları ve farklı churn etiketleri taşıyan özellik gruplarını ayrı ayrı inceler.

```python
# CustomerID ve Churn dışındaki sütunlar
feature_cols = [
    col for col in df.columns
    if col not in ["CustomerID", "Churn"]
]

# Özellikleri ve Churn etiketi aynı olan kayıtlar
duplicate_with_label = df.duplicated(
    subset=feature_cols + ["Churn"]
).sum()

# Özellikleri aynı olan kayıtlar
duplicate_features = df.duplicated(
    subset=feature_cols
).sum()

# Aynı özelliklere sahip ancak farklı Churn etiketleri taşıyan gruplar
label_counts = (
    df.groupby(feature_cols, dropna=False)["Churn"]
      .nunique()
)

conflicting_groups = (label_counts > 1).sum()

print("Tekrarlanan özellik + Churn kayıtları:", duplicate_with_label)
print("Tekrarlanan özellik kayıtları:", duplicate_features)
print("Farklı Churn etiketlerine sahip özellik grupları:", conflicting_groups)
```

**Yorumlama:**

- `duplicate_with_label`: Özellikleri ve churn etiketi aynı olan tekrarlanan kayıtları gösterir.
- `duplicate_features`: Churn etiketi dikkate alınmadan, aynı özelliklere sahip kayıtları gösterir.
- `conflicting_groups`: Aynı özelliklere sahip olmasına rağmen farklı churn etiketleri bulunan grupları gösterir.

Bu sonuçlar, kayıtların gerçekten hatalı olup olmadığını anlamak için incelenmelidir. Farklı müşteriler aynı özelliklere sahip olabilir. Bu nedenle, kayıtları doğrudan silmek yerine veri yapısı ve müşteri kimlikleri dikkate alınmalıdır.

#### 1.2. Eğitim ve Test Verilerinin Örtüşmesinin Kontrolü

Modelin başarısını doğru değerlendirebilmek için test verisinin eğitim verisinden bağımsız olması gerekir. Aynı özellik kombinasyonlarının iki bölümde de bulunması, model performansının olduğundan yüksek görünmesine neden olabilir.

```python
# Eğitim ve test verilerinde aynı özellik sütunlarını kullan
Xtr_check = Xtr[feature_cols]
Xte_check = Xte[feature_cols]

# Satırları karşılaştırılabilir anahtarlara dönüştür
train_keys = set(
    pd.util.hash_pandas_object(
        Xtr_check, index=False
    ).tolist()
)

test_keys = set(
    pd.util.hash_pandas_object(
        Xte_check, index=False
    ).tolist()
)

# Her iki bölümde bulunan özellik kombinasyonları
overlap = train_keys.intersection(test_keys)

print("Eğitim örneği sayısı:", len(Xtr_check))
print("Test örneği sayısı:", len(Xte_check))
print("Ortak özellik kombinasyonu sayısı:", len(overlap))
```

**Yorumlama:**

Ortak özellik kombinasyonlarının bulunması, aynı özelliklere sahip kayıtların eğitim ve test verilerinde yer aldığını gösterir. Bu durum otomatik olarak veri sızıntısı anlamına gelmez; ancak özellikle çok yüksek model skorları elde edildiğinde detaylı incelenmelidir.

Bu kontrol yalnızca özellik kombinasyonlarını karşılaştırır. Gerçek kayıt tekrarları ve olası veri sızıntıları ayrıca değerlendirilmelidir.

#### 1.3. Data Leakage Kontrolü

Data leakage, modelin tahmin sırasında gerçekte sahip olamayacağı bilgileri kullanarak eğitim veya değerlendirme sürecinde yapay olarak yüksek başarı elde etmesidir.

Modelde `Churn` hedef değişken, `CustomerID` ise kimlik bilgisi olduğu için özelliklerden çıkarılmalıdır.

```python
print("Churn, X içinde mi?", "Churn" in X.columns)
print("CustomerID, X içinde mi?", "CustomerID" in X.columns)

assert "Churn" not in X.columns
assert "CustomerID" not in X.columns
```

Her iki kontrolün sonucu da `False` olmalıdır.

Bununla birlikte, yalnızca bu iki sütunu çıkarmak yeterli değildir. Diğer değişkenlerin de tahmin anında bilinebilir olması gerekir.

| Değişken | Kontrol edilmesi gereken konu |
|---|---|
| `Tenure` | Müşterinin sistemde geçirdiği süre tahmin tarihinde biliniyor mu? |
| `Complain` | Şikâyet bilgisi tahmin tarihinden önce mi oluşmuş? |
| `DaySinceLastOrder` | Son sipariş bilgisi tahmin tarihinde mevcut mu? |
| `OrderCount` | Sipariş sayısı yalnızca tahmin tarihine kadarki verileri mi içeriyor? |
| `CashbackAmount` | Cashback bilgisi tahmin tarihine göre mi hesaplanmış? |

Bu değişkenler tek başına sızıntı göstergesi değildir. Değerlendirme, verinin nasıl ve ne zaman toplandığına bağlıdır.

### 2. Model Performansının Doğrulanması

Önceki modelleme aşamasında elde edilen sonuçlar aşağıdaki gibidir:

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.894 | 0.726 |
| Random Forest | 0.972 | 0.918 |
| XGBoost | 0.965 | 0.883 |

Random Forest ve XGBoost modellerinin yüksek skorları, churn müşterilerini ayırt etme konusunda başarılı olduklarını göstermektedir. Ancak bu performansın güvenilirliğini değerlendirmek için çapraz doğrulama ve farklı veri bölmeleriyle ek testler yapılmalıdır.

PR-AUC özellikle önemlidir; çünkü veri setinde churn oranı yaklaşık %16,8'dir ve hedef sınıflar dengeli değildir.

### 3. Feature Selection ve Ablation Analizi

Bu aşamada hangi özelliklerin model performansına katkı sağladığı incelenecektir.

- **Permutation Importance:** Bir özelliğin değerleri karıştırıldığında model başarısının ne kadar düştüğünü ölçer.
- **Feature Ablation:** Bir özellik veya özellik grubu çıkarılarak model yeniden eğitilir ve performans değişimi değerlendirilir.

Önceki analizde `Tenure`, `Complain` ve `NumberOfAddress` birden fazla önem ölçümünde öne çıkmıştır. Bununla birlikte, bir özelliğin önemli görünmesi nedensellik göstermez.

### 4. SHAP ile Modelin Açıklanması

SHAP, özelliklerin model tahminlerine katkısını açıklamak için kullanılır.

Bu aşamada:

- Genel özellik önem sıralaması incelenecektir.
- Özelliklerin churn tahminini hangi yönde etkilediği araştırılacaktır.
- Tek bir müşteri için tahminin neden yüksek veya düşük olduğu açıklanacaktır.

SHAP sonuçları modelin öğrendiği ilişkileri gösterir; tek başına nedensel bir açıklama sağlamaz.

### 5. İş Önerileri ve Sonuçların Raporlanması

Model doğrulandıktan sonra elde edilen bulgular iş aksiyonlarına dönüştürülecektir. Örneğin, yeni müşterilerin erken dönemde kaybedilmesi riski doğrulanırsa ilk 30–60 günlük müşteri deneyimi iyileştirilebilir. Şikâyet ve düşük cashback tutarıyla ilişkili churn riski de uygun analizlerle değerlendirilerek hedefli müşteri tutundurma çalışmaları tasarlanabilir.

Bu öneriler, analiz sonuçlarının doğrulanmasına bağlı hipotezler olarak ele alınmalıdır.

**İzlenecek sıra:**

1. Veri güvenilirliği ve data leakage kontrolü
2. Model performansının çapraz doğrulama ile değerlendirilmesi
3. Feature selection ve ablation analizi
4. SHAP ile modelin açıklanması
5. Bulguların raporlanması ve iş önerilerinin geliştirilmesi
