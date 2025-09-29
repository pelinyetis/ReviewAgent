# Smart Contract Legal Review Agent (TR)
**Akıllı Sözleşme Analiz Sistemi - Tüm Sözleşme Türleri**

## Kurulum

1) Bağımlılıklar
```bash
pip install -r requirements.txt
```

2) Ortam değişkenleri (.env veya sistem)
```
ANTHROPIC_API_KEY=...                    # Claude için zorunlu
MCP_SERVER_URL=https://platform.uipath.com/[account]/[service]/mcp/[server-name]  # UiPath MCP Server URL
UIPATH_ACCESS_TOKEN=...                  # UiPath authentication token (opsiyonel)
```

### MCP Server URL Formatı
UiPath Orchestrator'daki MCP server URL'inizi şu formatta kullanın:
```
https://platform.uipath.com/[HesapAdı]/[ServisAdı]/mcp/[MCP_Sunucu_Adı]
```

## Test Etme
MCP server bağlantısını test edin (Builder.py yaklaşımı):
```bash
python test_mcp_professional.py
```

## Çalıştırma
- **Akıllı Sözleşme Analizi**: Sözleşme türünü otomatik tespit eder ve uygun hukuki çerçeveyi uygular
- Agent, sözleşmeden türe özel hukuki konuları çıkarır (`extract_queries`), MCP'den ilgili mevzuatı çeker (`consult_mcp`) ve uzman rapor üretir (`generate_report`)
- **Desteklenen Türler**: Hizmet, ticari, kira, gizlilik, ortaklık sözleşmeleri ve daha fazlası
- **Adaptif Analiz**: Her sözleşme türü için uygun hukuki çerçeve ve mevzuat araştırması
- Çıktı olarak sadece PDF dosya yolu döner (`pdf_file_path`). PDF `Documents` klasörüne kaydedilir
- **MCP Entegrasyonu**: Boolean operator stratejileri ile optimize edilmiş mevzuat araştırması

## Notlar
- Türkçe karakterler için DejaVuSans/Arial font desteği vardır; sistemde mevcut değilse Helvetica’ya düşer.
- Bucket erişimi için UiPath Orchestrator ayarlarınızı doğrulayın (401 hatası için token/izin kontrolü).
