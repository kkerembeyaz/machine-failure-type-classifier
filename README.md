# Machine Failure Type Classifier

AI4I 2020 Predictive Maintenance veri seti üzerine kurulmuş, PyTorch ile eğitilmiş bir **multiclass classification** modeli ve bu modeli servis eden bir **FastAPI web uygulaması**. Proje, bir makinenin sensör verilerine (sıcaklık, devir, tork, takım aşınması, ürün kalite sınıfı) bakarak hiç arıza olup olmadığını, varsa hangi türden bir arıza olduğunu tahmin ediyor.

---

## İçindekiler

- [Proje Motivasyonu](#proje-motivasyonu)
- [Veri Seti](#veri-seti)
- [Veri Tutarlılık Kontrolü ve Temizlik](#veri-tutarlılık-kontrolü-ve-temizlik)
- [Hedef Etiketin Oluşturulması](#hedef-etiketin-oluşturulması)
- [Feature Engineering ve Encoding](#feature-engineering-ve-encoding)
- [Train / Validation / Test Split](#train--validation--test-split)
- [Scaling](#scaling)
- [Model Mimarisi](#model-mimarisi)
- [Class Imbalance Problemi ve 4 Model Denemesi](#class-imbalance-problemi-ve-4-model-denemesi)
- [Final Model: Test Seti Sonuçları](#final-model-test-seti-sonuçları)
- [Bilinen Limitasyonlar](#bilinen-limitasyonlar)
- [Model Kaydetme (state_dict)](#model-kaydetme-state_dict)
- [Web Uygulaması](#web-uygulaması)
- [Deployment Sürecinde Karşılaşılan Sorun](#deployment-sürecinde-karşılaşılan-sorun)
- [Nasıl Çalıştırılır](#nasıl-çalıştırılır)
- [Olası Gelecek İyileştirmeler](#olası-gelecek-iyileştirmeler)

---

## Proje Motivasyonu

Bu proje, bir üniversite ödevi kapsamında ("iris classifier benzeri bir multiclass classification web app") hazırlandı. Iris gibi çok klasik ve "tutorial seviyesi" bir veri seti yerine, gerçek bir endüstriyel problemi yansıtan, Bursa'daki sanayi firmalarının (Bosch, Tofaş, Siemens, Renault vb.) çalışma alanına yakın bir veri seti tercih edildi — bu, önceki bir ML dersi final projesinde (steel industry energy forecasting) kullanılan yaklaşımın devamı niteliğinde.

**Seçim kriterleri:** Multiclass bir hedef sunması, az input ile anlaşılır bir web app arayüzü mümkün kılması, ve gerçekçi bir "tahminsel bakım" (predictive maintenance) senaryosu barındırması.

## Veri Seti

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository), 10.000 satırlık sentetik bir üretim makinesi simülasyonu.

**Ham sütunlar:** `UDI`, `Product ID`, `Type` (L/M/H ürün kalite segmenti), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure` (genel arıza bayrağı), ve beş spesifik arıza türü bayrağı: `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

**Arıza türlerinin anlamı:**

| Kod | Ad | Tetiklenme koşulu |
|---|---|---|
| TWF | Tool Wear Failure | Takım aşınması ~200-240 dk aralığına ulaşınca |
| HDF | Heat Dissipation Failure | Hava-proses sıcaklık farkı düşük (~8.6K altı) + rpm ~1380 altı |
| PWF | Power Failure | Güç (tork × açısal hız) ~3500W altı ya da ~9000W üstü |
| OSF | Overstrain Failure | Takım aşınması × tork, ürün kalitesine (L/M/H) göre değişen bir eşiği aşarsa |
| RNF | Random Failure | Parametrelerden bağımsız, ~%0.1 olasılıkla rastgele |

`UDI` ve `Product ID` sadece kimlik bilgisi, modele hiç verilmedi.

## Veri Tutarlılık Kontrolü ve Temizlik

Dokümantasyonun varsaydığı kural ("Machine failure=1 ise mutlaka 5 türden en az biri 1'dir, ve asla birden fazlası aynı anda 1 olmaz") gerçek veride tam sağlanmıyordu. Bu yüzden iki tutarsızlık türü tespit edilip veri setinden çıkarıldı:

- **9 satır:** `Machine failure=1` ama 5 arıza türünün hepsi 0 (hangi türden olduğu belirsiz).
- **24 satır:** Birden fazla arıza türü aynı anda 1 (örtüşen arızalar — multiclass tanımı gereği bir örnek tek bir sınıfa ait olmalı, bu satırlar tanıma aykırı).

**Karar:** Bu 33 satır (%0.33) veri setinden çıkarıldı. Oran çok küçük olduğu için bu kaybın model performansına anlamlı bir etkisi beklenmiyor. Alternatif olarak bu satırlar için ayrı bir "Unknown/Multiple Failure" sınıfı açmak da düşünüldü, ancak örnek sayısının (özellikle 9'luk grup) bir sınıf olarak öğrenilemeyecek kadar az olması nedeniyle bu yoldan vazgeçildi.

## Hedef Etiketin Oluşturulması

5 binary arıza sütunu (TWF/HDF/PWF/OSF/RNF), `numpy.select()` kullanılarak tek bir `Failure_Type` sütununda (6 sınıflı, karşılıklı dışlayan bir etiket: No Failure, TWF, HDF, PWF, OSF, RNF) birleştirildi. Temizlik adımı sayesinde her satırda en fazla bir arıza türü işaretli kaldığı için, bu birleştirme (ilk True koşulu alma mantığı) güvenle uygulanabildi — temizlik yapılmasaydı, örtüşen satırlarda bir arıza türü sessizce kaybolurdu.

Not: Buradaki 5 sütun görünüşte one-hot encoding'e benziyor, ama aslında **bağımsız binary flag'lerdir** (multi-label yapı) — one-hot'un aksine birden fazlası aynı anda 1 olabiliyordu (örtüşen 24 satır bunun kanıtı). Proje hocanın istediği gibi (iris referanslı) **multiclass** olarak kurgulandığı için, çok etiketli (multi-label) bir mimari yerine tek etiketli (single-label, `CrossEntropyLoss` ile) bir yapı tercih edildi.

**Sonuç dağılımı (temizlik sonrası, 9967 satır):**

| Sınıf | Örnek Sayısı | Oran |
|---|---|---|
| No Failure | 9643 | %96.7 |
| HDF | 106 | %1.1 |
| PWF | 80 | %0.8 |
| OSF | 78 | %0.8 |
| TWF | 42 | %0.4 |
| RNF | 18 | %0.2 |

Bu ciddi dengesizlik, projenin en büyük teknik zorluğu oldu (bkz. [Class Imbalance](#class-imbalance-problemi-ve-4-model-denemesi)).

## Feature Engineering ve Encoding

- **Girdi özellikleri (`feature_cols`):** `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` — toplam 6 özellik.
- **`Type` sütunu — Ordinal Encoding (L=0, M=1, H=2):** One-hot yerine ordinal tercih edildi çünkü L/M/H arasında gerçek bir sıra var (düşük→yüksek kalite/maliyet segmenti). Bu, modele sıra bilgisini tek bir boyutta, doğrudan veriyor. Ayrıca `Type`, OSF kuralının (tork × aşınma eşiği ürün kalitesine göre değişiyor) doğrudan bir parçası olduğu için, "önemsiz görünse bile" modelden çıkarılmadı — hangi özelliğin önemli olduğuna modelin kendisinin karar vermesine izin verildi.
- **Hedef (`Failure_Type`) — Manuel mapping:** `LabelEncoder`'ın alfabetik sıralaması yerine elle tanımlanan bir sözlük (`No Failure`=0, `TWF`=1, ... `RNF`=5) kullanıldı — hem okunabilirlik hem de web app'te tahmin sonucunu (class index) tekrar isme çevirirken tutarlılık için.

## Train / Validation / Test Split

Stratified split, 70/15/15 oranında (`train_test_split` iki aşamalı uygulanarak): **6976 / 1495 / 1496** satır.

`stratify` kullanımı özellikle kritikti — RNF gibi sadece 18 örneği olan bir sınıf, rastgele bölmede tamamen bir sete düşüp diğerlerinde hiç temsil edilmeyebilirdi. Stratified split, her sınıfın oranını train/val/test'in her birinde korudu.

## Scaling

5 sayısal sütun (`Type` hariç) `StandardScaler` ile scale edildi. **Leakage'ı önlemek için:** scaler sadece `X_train`'e `fit_transform` edildi, `X_val`/`X_test`'e ise öğrenilen istatistiklerle sadece `transform` uygulandı. `Type` sütunu scale edilmedi çünkü zaten küçük, ordinal bir aralıkta (0,1,2).

## Model Mimarisi

Basit bir feed-forward ANN (`FailureClassifier`):

```
Linear(6, 16) → ReLU → Linear(16, 16) → ReLU → Linear(16, 6)
```

- Girdi boyutu (6) = özellik sayısı, çıktı boyutu (6) = sınıf sayısı.
- Çıkışa `softmax` eklenmedi çünkü `CrossEntropyLoss` bunu içeride zaten uyguluyor (çift softmax hatasından kaçınmak için).
- Optimizer: Adam, `lr=0.001`.
- Eğitim, veri boyutunun (6976 satır) küçük olması nedeniyle **batch'leme yapılmadan**, tüm train verisiyle her epoch'ta tek bir forward/backward geçişi şeklinde yürütüldü (batch gradient descent) — DataLoader/mini-batch yaklaşımı bilinçli olarak, kavramsal karmaşıklığı artırmamak için bu projede tercih edilmedi.

## Class Imbalance Problemi ve 4 Model Denemesi

Veri setinin %96.7'sinin tek bir sınıfa (No Failure) ait olması, projenin merkezi zorluğuydu. Bu problemi ele almak için dört farklı strateji sistematik olarak denendi ve karşılaştırıldı.

### Model 1 — Weight'siz eğitim (baseline)

Hiçbir dengesizlik önlemi alınmadan eğitildi. **Sonuç:** Model, train setindeki 6976 örneğin **tamamına** `"No Failure"` tahmini verdi (`torch.unique(pred)` → tek bir değer). Macro accuracy ~%16-17'de sabit kaldı — bu, 6 sınıf için "sadece çoğunluk sınıfını tahmin eden" bir modelin ulaşabileceği matematiksel tavan (`1/6 ≈ %16.7`).

**Neden loss düşüyordu ama accuracy sabitti:** `CrossEntropyLoss`, modelin tahminine olan güvenini de ölçüyor. Model, No Failure örneklerindeki tahminine giderek daha "emin" oldukça (olasılığı %60'tan %99'a çıkardıkça) loss düşüyordu — ama bu, azınlık sınıfları hiç öğrenmediği gerçeğini değiştirmiyordu. Accuracy (nihai karar, argmax) bu "emin olma" sürecini yakalamıyor, sadece doğru/yanlışı ölçüyordu.

### Model 2 — `compute_class_weight(balanced)` ile eğitim

Sklearn'in `'balanced'` formülüyle hesaplanan ağırlıklar kullanıldı: `[0.17, 40.09, 15.71, 20.76, 21.14, 89.44]` (No Failure → RNF sırasıyla). Az örnekli sınıflara orantılı olarak çok yüksek ceza verildi.

**Sonuç:** Azınlık sınıfların recall'u belirgin şekilde iyileşti (TWF ~%83, HDF ~%94, OSF ~%91), ama bu sefer **No Failure precision'ı ~%50'ye düştü** — yani model artık No Failure örneklerinin yarısında yanlışlıkla bir arıza türü tahmin ediyordu. Ağırlıklar o kadar agresifti ki model "riske girip azınlık sınıf de" stratejisine aşırı kaydı.

### Model 3 — Tavanlı (capped) ağırlıklar

Model 2'deki ağırlıklara bir üst sınır (`np.minimum(weights, 10.0)`) uygulanarak aşırılık yumuşatıldı.

**Sonuç:** No Failure precision ~%99'a çıktı (beklenen iyileşme), ama bu sefer **TWF ve RNF recall'u %0'a düştü** — tavan, bu iki sınıfın ağırlığını (özellikle TWF'nin 40.09'dan 10'a) o kadar düşürdü ki model onları öğrenmeye değer bulmadı. Yeni bir trade-off ortaya çıktı: No Failure'ı kazanırken TWF'yi kaybettik.

### Model 4 — Manuel, sınıf bazlı ağırlıklar (FİNAL MODEL)

Model 2 ve Model 3'ün sonuçları karşılaştırılarak, her sınıfa ayrı ayrı, gerekçeli bir ağırlık elle belirlendi:

| Sınıf | Ağırlık | Gerekçe |
|---|---|---|
| No Failure | 0.5 | Model 3'te 0.17 çok düşüktü (precision feda ediliyordu); hafif artırıldı |
| TWF | 20.0 | Model 3'te tavan (10) yetersiz kaldı (recall %0); orta bir değere çekildi |
| HDF | 10.0 | Model 3'te zaten iyi çalışıyordu (recall %94); değiştirilmedi |
| PWF | 12.0 | Hafif artırıldı |
| OSF | 12.0 | Model 3'te iyi çalışıyordu (recall %91); korundu |
| RNF | 15.0 | Tanım gereği (parametrelerden bağımsız, rastgele) öğrenilemez; makul bir değerde bırakıldı |

**Sonuç (validation):** No Failure recall %91.5'e, precision ~%99'a çıktı (ikisi birden iyileşti); TWF recall kısmen geri geldi (%33); HDF recall %68.75'e geriledi (Model 3'e göre düşüş, ama kabul edilebilir); OSF recall %90.9 korundu; RNF recall %0 (beklenen, bkz. limitasyonlar).

### Üç modelin karşılaştırması

| Metrik | Model 2 (balanced) | Model 3 (tavan=10) | Model 4 (manuel, final) |
|---|---|---|---|
| No Failure precision | ~%50 | ~%99 | ~%99 |
| No Failure recall | yüksek | %80.4 | **%91.5** |
| TWF recall | %83 | **%0** | %33 |
| HDF recall | iyi | %93.75 | %68.75 |
| OSF recall | iyi | %90.9 | %90.9 |
| RNF recall | %0 | %0 | %0 |

**Karar:** Model 4 final model olarak seçildi. Gerekçe: hiçbir tek ağırlık stratejisi tüm sınıflarda mükemmel sonuç vermiyor (class imbalance bu kadar şiddetliyken beklenen bir durum); Model 4, en kritik iki metriği (No Failure precision ve recall) güçlü tutarken diğer sınıfları da tamamen terk etmiyor. Daha fazla deneme yapmak yerine burada durulmasının nedeni, azalan getiri noktasına ulaşılmış olması — üç ardışık deneme de benzer trade-off'lar sergiledi, bu paternin dördüncü/beşinci denemede kırılacağına dair bir işaret yoktu.

## Final Model: Test Seti Sonuçları

Test seti, model seçimi tamamlanana kadar **hiç kullanılmadı** — sadece Model 4 belirlendikten sonra, **tek seferlik** olarak değerlendirildi (bu, val set üzerinde yapılan tüm karşılaştırmalı denemelerin test setine "sızmasını" önlemek için bilinçli bir karardı).

Test sonuçları, validation sonuçlarıyla büyük ölçüde tutarlı çıktı: No Failure precision ~%99, OSF recall %100, HDF recall %80, PWF recall %30. TWF'de val'de görülen kısmi iyileşme (%33 recall) test setinde görülmedi (recall %0) — bu farkın nedeni örnek sayısının çok küçük olması (test setinde sadece 7 TWF örneği var, tek bir model kararı bile yüzdeyi tamamen değiştiriyor).

## Bilinen Limitasyonlar

- **RNF sınıfı tanım gereği öğrenilemez.** AI4I dokümantasyonuna göre RNF, girdi parametrelerinden tamamen bağımsız, ~%0.1 olasılıkla oluşan rastgele bir arıza. Hiçbir class weight stratejisi bunu değiştiremez; recall %0 kalması bir model hatası değil, verinin doğasından kaynaklanan bir sınırlılıktır.
- **Küçük örnekli sınıflarda test sonuçları değişken/güvenilmezdir.** RNF (test setinde ~3 örnek) ve TWF (test setinde 7 örnek) gibi sınıflarda, tek bir yanlış tahmin bile recall'u büyük oranda değiştirir. Bu sınıflardaki sonuçlar temkinli yorumlanmalıdır.
- **33 satır (unknown + overlap) veri setinden çıkarıldı** — veri temizliği kapsamında, oran küçük olduğu için etkisi ihmal edilebilir kabul edildi.
- **Class weight ayarı elle, deneme-yanılma yoluyla yapıldı.** Daha sistematik yöntemler (örneğin SMOTE ile oversampling, focal loss gibi dengesizliğe özel loss fonksiyonları) bu projenin kapsamı dışında bırakıldı; bkz. [Gelecek İyileştirmeler](#olası-gelecek-iyileştirmeler).
- **`hidden_dim=16` ile sabit kalındı**, eğitim sonuçlarına göre 32'ye çıkarma opsiyonu düşünüldü ama mevcut sonuçlar yeterli görüldüğü için uygulanmadı.

## Model Kaydetme (state_dict)

Sadece model ağırlıkları (`state_dict`) değil, inference için gereken tüm yardımcı nesneler `torch.save`/`torch.load` ile kaydedildi — çünkü `state_dict` mimariyi, scaler'ı ya da label mapping'lerini tutmuyor:

```
models/
├── machine-failure-classifier.pth   # model ağırlıkları (state_dict)
├── scaler.pth                       # StandardScaler (fit edilmiş haliyle)
├── type_order.pth                   # {'L':0,'M':1,'H':2}
├── label_map.pth                    # string -> class index
└── label_map_inv.pth                # class index -> string
```

Geri yüklerken, model mimarisinin (`FailureClassifier` sınıfı) `state_dict`'i yüklemeden önce tanımlı olması gerekiyor — `state_dict` sadece sayıları tutar, katman yapısını değil.

## Web Uygulaması

FastAPI ile geliştirildi (kod üretimi için Antigravity kullanıldı, model entegrasyonu ve doğrulaması elle yapıldı). Arayüz, ayrı HTML/CSS/JS dosyaları yerine `HTMLResponse` ile Python içine gömülü — küçük ölçekli bir proje için geçerli ve basit bir tercih.

**`/predict` endpoint'inin preprocessing sırası (eğitimdekiyle birebir aynı olmalı):**

1. `Type` → `type_order` ile encode et (L=0, M=1, H=2)
2. 5 sayısal alanı → yüklenen `scaler.transform()` ile scale et (fit değil, sadece transform)
3. Sütun sırası: `[Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear]`
4. `torch.tensor(..., dtype=torch.float32)` → model (`eval()` modunda, `inference_mode()` içinde) → `argmax` → `label_map_inv` ile isme çevir

**Doğrulama yöntemi:** Notebook içinde `manual_predict()` adında, web app'in preprocessing mantığını birebir taklit eden bir fonksiyon yazıldı. Aynı girdi hem bu fonksiyona hem web app'e verilip, iki çıktının birebir eşleştiği teyit edildi.

## Deployment Sürecinde Karşılaşılan Sorun

Doğrulama sırasında, **aynı girdi için notebook ve web app'in farklı sonuçlar verdiği** tespit edildi (notebook: "No Failure", web app: "PWF"). Kod incelemesi ve karşılaştırmalı testler sonunda kaynağı bulundu:

- **Scaler tutarlıydı** çünkü `train_test_split(random_state=42)` sabit bir seed kullanıyordu — her script çalıştırmasında `X_train` hep aynı satırlardan oluştuğu için `scaler.fit()` hep aynı `mean_`/`scale_` değerlerini üretiyordu.
- **Model tutarsızdı** çünkü `FailureClassifier()` çağrıldığında ağırlıklar PyTorch tarafından rastgele başlatılıyordu ve script'te bunu sabitleyen bir `torch.manual_seed()` yoktu. Script her tekrar çalıştırıldığında, kaydedilen `.pth` dosyası aslında **farklı bir model4**'ü temsil ediyordu.
- **Asıl tetikleyici:** FastAPI sunucusu, modeli sadece **açılışta bir kez** belleğe yüklüyor. Script'in tekrar çalıştırılıp `models/` klasöründeki dosyaların üzerine yazılması, zaten çalışmakta olan sunucudaki bellek içi modeli **güncellemedi** — sunucu, saatler önce yüklediği eski ağırlıklarla cevap vermeye devam etti. `--reload` seçeneği yalnızca `.py` dosyalarındaki değişiklikleri izliyor, `.pth` dosyalarındaki değişiklikleri fark etmiyor.

**Çözüm ve alınan dersler:**

1. Script'in başına `torch.manual_seed(42)` eklendi — böylece model artık her çalıştırmada aynı, tekrarlanabilir ağırlıklarla başlıyor.
2. Model dosyaları her güncellendiğinde, FastAPI sunucusunun **manuel olarak yeniden başlatılması** gerektiği not edildi — `--reload` bu konuda yardımcı olmuyor.
3. Bu, "model eğittim, kaydettim, bitti" sanılan bir adımın aslında gerçek bir production/deployment sorunu barındırabileceğinin somut bir örneği oldu — model dosyasının doğru olması yetmiyor, servisin o dosyayı **ne zaman ve nasıl** okuduğunu da bilmek gerekiyor.

## Nasıl Çalıştırılır

```bash
# Modeli eğitmek için
python machine-failure-type-classifier.py

# Web uygulamasını başlatmak için
uvicorn app:app --reload
# tarayıcıda http://127.0.0.1:8000/ adresine git
```

> Not: Model dosyalarını (`models/` klasörü) güncelledikten sonra, sunucuyu mutlaka yeniden başlatın — bkz. [Deployment Sürecinde Karşılaşılan Sorun](#deployment-sürecinde-karşılaşılan-sorun).

## Olası Gelecek İyileştirmeler

- Class weight'leri elle ayarlamak yerine, **SMOTE** gibi oversampling tekniklerini ya da **focal loss** gibi dengesizliğe özel loss fonksiyonlarını denemek.
- `hidden_dim`'i artırıp (örneğin 32) daha fazla epoch ile eğitip, val/test loss eğrisinin plato yaptığı noktayı (ve varsa overfitting başlangıcını) daha sistematik incelemek.
- RNF dışındaki sınıflar için confusion matrix'teki en çok karışan çiftleri (örneğin HDF↔OSF gibi) ayrıca inceleyip, bu karışıklığı azaltacak ek feature engineering (örneğin dokümantasyondaki kurallara daha yakın türetilmiş özellikler: sıcaklık farkı, güç, tork×aşınma) denemek.
- Web app'e, modelin tahminine ne kadar "emin" olduğunu (olasılık dağılımını) kullanıcıya daha görsel şekilde (progress bar, renk kodlama) sunmak.
