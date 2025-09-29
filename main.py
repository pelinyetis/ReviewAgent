import logging
import os
from typing import Optional, Dict, List, Any
import json
import traceback
import asyncio
from dotenv import load_dotenv
import re
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
import PyPDF2
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

from uipath import UiPath
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Environment variables - NO HARDCODED PATHS
MCP_SERVER_URL = os.getenv("UIPATH_MCP_SERVER_URL")
UIPATH_ACCESS_TOKEN = os.getenv("UIPATH_ACCESS_TOKEN") 
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
UIPATH_BUCKET_NAME = os.getenv("UIPATH_BUCKET_NAME", "SB_Contracts")  # Can be overridden
UIPATH_FOLDER_PATH = os.getenv("UIPATH_FOLDER_PATH", "Shared")  # Can be overridden
BASE_DOCUMENT_PATH = os.getenv("BASE_DOCUMENT_PATH", os.path.join(os.getcwd(), "Documents"))  # Dynamic base path

# Configuration
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "7"))  # Increased for better research
MAX_TOOL_RESPONSE_LENGTH = int(os.getenv("MAX_TOOL_RESPONSE_LENGTH", "4000"))
MAX_CONTRACT_LENGTH = int(os.getenv("MAX_CONTRACT_LENGTH", "3000"))

# Font setup for Turkish characters
def setup_turkish_font():
    """Setup font with Turkish character support"""
    font_options = [
        ('DejaVuSans', 'DejaVuSans.ttf'),
        ('Arial', 'arial.ttf'),
        ('FreeSans', 'FreeSans.ttf')
    ]
    
    for font_name, font_file in font_options:
        try:
            pdfmetrics.registerFont(TTFont(font_name, font_file))
            logger.info(f"Using {font_name} font for Turkish characters")
            return font_name
        except:
            continue
            
    logger.warning("No Turkish font found, using Helvetica")
    return 'Helvetica'

TURKISH_FONT = setup_turkish_font()

# UiPath client
uipath = UiPath()

# Pydantic models
class GraphInput(BaseModel):
    document_name: str
    bucket_name: Optional[str] = None
    folder_path: Optional[str] = None
    output_path: Optional[str] = None

class GraphOutput(BaseModel):
    pdf_file_path: str
    analysis_summary: Optional[str] = None
    
class GraphState(MessagesState):
    document_name: str
    contract_text: str
    report_text: str
    extracted_values: Dict[str, Any]
    bucket_name: str
    folder_path: str
    output_path: str

def extract_contract_values(contract_text: str) -> Dict[str, Any]:
    """Dynamically extract ALL identifiable values from contract text"""
    extracted = {}
    
    # Define extraction patterns dynamically
    extraction_patterns = {
        "parties": [
            r'(?:Taraf|Alıcı|Satıcı|Müşteri|Tedarikçi|İşveren|İşçi|Kiracı|Kiraya veren|Vekil|Müvekkil)[\s:]+([A-ZĞÜŞİÖÇ][^\n\.]+)',
            r'(?:bir taraftan|diğer taraftan)\s+([A-ZĞÜŞİÖÇ][^\n,]+)',
            r'(?:Firma|Şirket|Limited|A\.Ş\.|Ltd\.|Tic\.|LTD|ŞTİ)[\s:]*([A-ZĞÜŞİÖÇ][^\n\.]+)',
            r'(?:T\.C\.|TC)\s+(?:Kimlik No|VKN)[\s:]+(\d{10,11})',
        ],
        "dates": [
            r'(\d{1,2}[\.\/]\d{1,2}[\.\/]\d{4})',
            r'(\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})',
            r'(?:Tarih|İmza tarihi|Sözleşme tarihi|Başlangıç|Bitiş)[\s:]+([^\n]+)',
        ],
        "durations": [
            r'(?:süre|vade|dönem|müddet)[\s:]*(\d+)\s*(?:gün|ay|yıl|hafta|saat)',
            r'(\d+)\s*(?:günlük|aylık|yıllık|haftalık)\s+(?:süre|vade)',
            r'(?:başlangıç|bitiş)\s+(?:tarihi|zamanı)[\s:]+([^\n]+)',
        ],
        "amounts": [
            r'([\d\.]+(?:\.\d{3})*(?:,\d{2})?)\s*(?:TL|TRY|USD|EUR|GBP|₺|\$|€|£)',
            r'(?:tutar|bedel|ücret|fiyat|maliyet|değer)[\s:]+([^\n]+)',
            r'(?:toplam|ara toplam|KDV|ÖTV|vergi)[\s:]+([^\n]+)',
            r'(?:peşinat|kaparo|avans|depozito)[\s:]+([^\n]+)',
        ],
        "percentages": [
            r'(%\s*\d+(?:[,\.]\d+)?)',
            r'(yüzde\s+\w+)',
            r'(\d+(?:[,\.]\d+)?)\s*(?:faiz|oran|indirim)',
        ],
        "payment_terms": [
            r'(?:ödeme|tahsilat|taksit|vade)[^\n.]{0,200}',
            r'(?:peşin|vadeli|kredi kartı|havale|EFT|çek|senet)[^\n.]{0,150}',
        ],
        "penalties": [
            r'(?:ceza|tazminat|gecikme|temerrüt)[^\n.]{0,200}',
            r'(?:cayma|fesih)\s+(?:bedeli|tazminatı|cezası)[^\n.]{0,150}',
        ],
        "termination": [
            r'(?:fesih|sona erme|iptal|tasfiye)[^\n.]{0,200}',
            r'(?:haklı|haksız)\s+(?:fesih|sebepler|nedenler)[^\n.]{0,150}',
        ],
        "warranties": [
            r'(?:garanti|teminat|kefalet|taahhüt)[^\n.]{0,200}',
            r'(?:ayıp|kusur|hasar|zarar)\s+(?:sorumluluğu|garantisi)[^\n.]{0,150}',
        ],
        "obligations": [
            r'(?:yükümlülük|sorumluluk|borç|vecibe)[^\n.]{0,200}',
            r'(?:taahhüt|edim|yüküm)[^\n.]{0,150}',
        ],
        "clauses": [
            r'(?:Madde|MADDE)\s+(\d+)[\s\-:]*([^\n]{0,300})',
        ],
        "legal_references": [
            r'(\d{4})\s*(?:sayılı|no\.lu|numaralı)\s+([^,\.\n]+)',
            r'(?:TBK|TTK|TMK|HMK|İİK|TCK)\s*(?:m\.|madde|md\.)\s*(\d+)',
        ],
        "addresses": [
            r'(?:Adres|İkamet|Merkez)[\s:]+([^\n]+)',
            r'(?:Mahalle|Mah\.|Cadde|Cad\.|Sokak|Sok\.)[^\n]{0,100}',
        ],
        "contact_info": [
            r'(?:Tel|Telefon|Gsm|Cep)[\s:]+([^\n]+)',
            r'(?:E-posta|Email|E-mail|Eposta)[\s:]+([^\n]+)',
            r'(?:Faks|Fax)[\s:]+([^\n]+)',
        ]
    }
    
    # Extract all patterns
    for category, patterns in extraction_patterns.items():
        extracted[category] = []
        for pattern in patterns:
            matches = re.findall(pattern, contract_text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle multi-group matches
                    extracted[category].append(' '.join([str(m) for m in match if m]))
                else:
                    extracted[category].append(str(match))
        
        # Clean and deduplicate
        if extracted[category]:
            # Remove duplicates while preserving order
            seen = set()
            unique = []
            for item in extracted[category]:
                item_clean = item.strip()[:300]  # Limit length
                if item_clean and item_clean not in seen:
                    seen.add(item_clean)
                    unique.append(item_clean)
            extracted[category] = unique[:10]  # Limit to 10 items per category
        else:
            del extracted[category]  # Remove empty categories
    
    # Add contract statistics
    extracted["statistics"] = {
        "total_length": len(contract_text),
        "word_count": len(contract_text.split()),
        "paragraph_count": len([p for p in contract_text.split('\n\n') if p.strip()]),
        "categories_found": len(extracted)
    }
    
    return extracted

async def prepare_input(state: GraphState) -> Command:
    """Prepare input by downloading and extracting text from PDF"""
    graph_input = GraphInput(document_name=state["document_name"])
    
    # Use environment variables or provided values
    bucket_name = graph_input.bucket_name or UIPATH_BUCKET_NAME
    folder_path = graph_input.folder_path or UIPATH_FOLDER_PATH
    output_path = graph_input.output_path or BASE_DOCUMENT_PATH
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Download PDF from UiPath
    file_name = f"{graph_input.document_name}.pdf"
    pdf_path = os.path.join(output_path, file_name)
    
    try:
        uipath.buckets.download(
            name=bucket_name,
            blob_file_path=file_name,
            destination_path=pdf_path,
            folder_path=folder_path
        )
        logger.info(f"Downloaded PDF to: {pdf_path}")
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        raise

    # Extract text from PDF
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except Exception as e:
        logger.error(f"Failed to read PDF: {e}")
        raise
    
    # Extract contract values dynamically
    extracted_values = extract_contract_values(text)
    logger.info(f"Extracted {len(extracted_values)} categories with {sum(len(v) if isinstance(v, list) else 1 for v in extracted_values.values())} total values")

    return Command(update={
        "document_name": graph_input.document_name,
        "contract_text": text,
        "extracted_values": extracted_values,
        "bucket_name": bucket_name,
        "folder_path": folder_path,
        "output_path": output_path,
        "messages": [HumanMessage(content=f"Contract loaded: {len(text)} characters, {len(extracted_values)} value categories extracted")]
    })

async def analyze_and_generate_report(state: GraphState) -> Command:
    """Analyze contract using LLM with optional MCP tools"""
    
    contract_text = state.get("contract_text", "")
    document_name = state.get("document_name", "unknown")
    extracted_values = state.get("extracted_values", {})
    output_path = state.get("output_path", BASE_DOCUMENT_PATH)
    
    if not contract_text:
        return Command(update={"report_text": "Sözleşme metni bulunamadı"})
    
    # Initialize LLM
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=ANTHROPIC_API_KEY,
        max_tokens=8000,
        temperature=0.1
    )
    
    # Try to load MCP tools if available
    mcp_tools = []
    mcp_available = False
    
    try:
        if MCP_SERVER_URL and UIPATH_ACCESS_TOKEN:
            logger.info("MCP server credentials found, attempting connection...")
            
            async with streamablehttp_client(
                url=MCP_SERVER_URL,
                headers={"Authorization": f"Bearer {UIPATH_ACCESS_TOKEN}"},
                timeout=30,
            ) as (read, write, session_id_callback):
                
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("MCP session initialized")
                    
                    mcp_tools = await load_mcp_tools(session)
                    mcp_available = len(mcp_tools) > 0
                    logger.info(f"MCP tools available: {mcp_available} ({len(mcp_tools)} tools)")
                    
                    if mcp_available:
                        report_text = await perform_analysis_with_mcp(
                            llm, contract_text, document_name, mcp_tools, extracted_values
                        )
                    else:
                        report_text = await perform_analysis_without_mcp(
                            llm, contract_text, document_name, extracted_values
                        )
        else:
            logger.info("No MCP credentials, proceeding without MCP tools")
            report_text = await perform_analysis_without_mcp(
                llm, contract_text, document_name, extracted_values
            )
            
    except Exception as e:
        logger.warning(f"MCP connection failed: {e}, proceeding without tools")
        report_text = await perform_analysis_without_mcp(
            llm, contract_text, document_name, extracted_values
        )
    
    if report_text:
        # Save PDF report
        pdf_output_path = os.path.join(output_path, f"{document_name}_Rapor.pdf")
        _save_report_as_pdf(report_text, pdf_output_path)
        logger.info(f"Report saved: {pdf_output_path}")
        
        return Command(update={"report_text": report_text})
    else:
        return Command(update={"report_text": "Rapor oluşturulamadı"})


async def perform_analysis_with_mcp(llm, contract_text, document_name, mcp_tools, extracted_values):
    """Perform analysis using MCP tools - fully dynamic"""
    try:
        logger.info("Starting MCP-enhanced analysis...")
        
        # Convert extracted values to JSON
        extracted_json = json.dumps(extracted_values, ensure_ascii=False, indent=2)
        
        # Dynamic system prompt - let LLM decide everything
        system_prompt = """Sen Türk hukuku konusunda uzman bir hukuk müşavirisin. Türk mevzuat veritabanına erişimin var.

MEVZUAT ARAÇLARI:
Aşağıdaki araçları kullanarak Türk mevzuatını araştırabilirsin:
- search_mevzuat: Anahtar kelimelerle mevzuat ara
- get_mevzuat_article_tree: Kanunun madde yapısını gör
- get_mevzuat_article_content: Madde içeriğini oku

ANALİZ SÜRECİ:
1. Sözleşmeyi incele, türünü ve kritik noktalarını belirle
2. Hangi kanunların uygulanacağına karar ver
3. İlgili mevzuatı araştır (araçları kullan)
4. Sözleşmeden çıkarılan değerleri kullanarak detaylı analiz yap
5. Riskleri ve önerileri belirle

Sözleşmenin gerektirdiği derinlikte araştırma yap. Neyin araştırılacağına, hangi maddelerin inceleneceğine sen karar ver.

RAPOR YAZIMI:
- Akıcı paragraflar halinde yaz
- Çıkarılan değerleri (tutarlar, tarihler, oranlar) **kalın** olarak vurgula
- Araştırdığın kanun maddelerini doğrudan referans ver
- Somut riskler ve çözüm önerileri sun"""

        user_prompt = f"""Bu sözleşmeyi analiz et:

ÇIKARILAN DEĞERLER:
{extracted_json[:3000]}

SÖZLEŞME METNİ:
{contract_text[:MAX_CONTRACT_LENGTH]}

Sözleşme türüne göre gerekli mevzuat araştırmasını yap ve kapsamlı hukuki analiz hazırla."""

        # Let LLM freely use tools
        llm_with_tools = llm.bind_tools(mcp_tools)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        iteration = 0
        report_text = ""
        
        while iteration < MAX_TOOL_ITERATIONS:
            try:
                response = await llm_with_tools.ainvoke(messages)
                messages.append(response)
                
                if not response.tool_calls:
                    report_text = response.content
                    break
                
                # Process tool calls
                for tool_call in response.tool_calls:
                    tool = next((t for t in mcp_tools if t.name == tool_call["name"]), None)
                    
                    if tool:
                        try:
                            tool_result = await tool.ainvoke(tool_call["args"])
                            
                            # Convert result to string
                            if isinstance(tool_result, (dict, list)):
                                tool_content = json.dumps(tool_result, ensure_ascii=False)
                                if len(tool_content) > MAX_TOOL_RESPONSE_LENGTH:
                                    tool_content = tool_content[:MAX_TOOL_RESPONSE_LENGTH] + "..."
                            else:
                                tool_content = str(tool_result)[:MAX_TOOL_RESPONSE_LENGTH]
                            
                            messages.append(ToolMessage(
                                content=tool_content,
                                tool_call_id=tool_call["id"]
                            ))
                            
                        except Exception as e:
                            logger.error(f"Tool error: {e}")
                            messages.append(ToolMessage(
                                content=f"Hata: {str(e)[:100]}",
                                tool_call_id=tool_call["id"]
                            ))
                
                iteration += 1
                
            except Exception as e:
                logger.error(f"Iteration error: {e}")
                break
        
        # If no report yet, request final report
        if not report_text:
            final_prompt = "Araştırmaları tamamla ve Türkçe hukuki analiz raporunu hazırla. Çıkarılan değerleri kullan."
            final_response = await llm.ainvoke(messages + [HumanMessage(content=final_prompt)])
            report_text = final_response.content
        
        return report_text
        
    except Exception as e:
        logger.error(f"MCP analysis failed: {e}")
        raise


async def perform_analysis_without_mcp(llm, contract_text, document_name, extracted_values):
    """Perform analysis without MCP tools - fully dynamic"""
    try:
        logger.info("Starting analysis without MCP tools...")
        
        # Convert extracted values to JSON
        extracted_json = json.dumps(extracted_values, ensure_ascii=False, indent=2)
        
        # Dynamic prompt
        system_prompt = """Sen deneyimli bir Türk hukuk müşavirisin. Sözleşmeleri detaylı analiz edip hukuki görüş hazırlıyorsun.

Mevzuat veritabanına erişimin olmadığı için genel hukuk bilginle analiz yapacaksın.

ANALİZ YÖNTEMİ:
1. Sözleşme türünü ve tarafları belirle
2. Kritik hükümleri tespit et
3. İlgili Türk hukuku kurallarını uygula
4. Risk ve uyumsuzlukları belirle
5. Somut öneriler sun

Sözleşmeden çıkarılan tüm değerleri (tutarlar, tarihler, oranlar) **kalın** olarak vurgulayarak kullan.

Raporu sözleşmenin gerektirdiği derinlik ve kapsamda, akıcı paragraflar halinde yaz."""

        user_prompt = f"""Bu sözleşmeyi analiz et:

ÇIKARILAN DEĞERLER:
{extracted_json[:3000]}

SÖZLEŞME METNİ:
{contract_text[:MAX_CONTRACT_LENGTH]}

Sözleşme türüne uygun kapsamlı hukuki analiz hazırla. Çıkarılan değerleri mutlaka kullan."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        
        # Add disclaimer for non-MCP analysis
        report_text = """⚠️ NOT: Bu analiz güncel mevzuat veritabanına erişim olmadan hazırlanmıştır. Kesin hukuki görüş için güncel mevzuat kontrolü önerilir.

═════════════════════════════════════════════

""" + response.content
        
        return report_text
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return f"Analiz hatası: {str(e)}"


def _save_report_as_pdf(text: str, output_path: str):
    """Save report as PDF with Turkish character support"""
    try:
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        styles = getSampleStyleSheet()
        
        style = ParagraphStyle(
            'Turkish',
            parent=styles['Normal'],
            fontName=TURKISH_FONT,
            fontSize=11,
            leading=14,
            spaceAfter=6
        )
        
        content = []
        for paragraph in text.split('\n'):
            if paragraph.strip():
                # Escape special characters and handle bold text
                paragraph = paragraph.replace('&', '&amp;')
                paragraph = paragraph.replace('<', '&lt;')
                paragraph = paragraph.replace('>', '&gt;')
                # Convert markdown bold to ReportLab bold
                paragraph = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', paragraph)
                
                content.append(Paragraph(paragraph, style))
            else:
                content.append(Spacer(1, 0.2*inch))
        
        doc.build(content)
        logger.info(f"PDF created successfully: {output_path}")
        
    except Exception as e:
        logger.error(f"PDF creation failed: {e}")
        # Fallback to text file
        txt_path = output_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Saved as text file: {txt_path}")


async def output_node(state: GraphState) -> GraphOutput:
    """Final output node"""
    output_path = state.get("output_path", BASE_DOCUMENT_PATH)
    document_name = state.get("document_name", "rapor")
    pdf_path = os.path.join(output_path, f"{document_name}_Rapor.pdf")
    
    # Extract summary from report
    report_text = state.get("report_text", "")
    summary = report_text[:500] + "..." if len(report_text) > 500 else report_text
    
    return GraphOutput(
        pdf_file_path=pdf_path,
        analysis_summary=summary
    )


# Build Graph
def build_graph():
    """Build the optimized LangGraph workflow"""
    builder = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)
    
    # Add nodes
    builder.add_node("prepare_input", prepare_input)
    builder.add_node("analyze_and_generate_report", analyze_and_generate_report)
    builder.add_node("output_node", output_node)
    
    # Add edges
    builder.add_edge(START, "prepare_input")
    builder.add_edge("prepare_input", "analyze_and_generate_report")
    builder.add_edge("analyze_and_generate_report", "output_node")
    builder.add_edge("output_node", END)
    
    # Compile with memory
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Create graph instance
graph = build_graph()

# Example usage
async def main():
    """Example usage of the contract analysis system"""
    try:
        # Initialize graph
        analysis_graph = build_graph()
        
        # Example input
        input_data = {
            "document_name": "sample_contract",
            # Optional: override defaults
            # "bucket_name": "CustomBucket",
            # "folder_path": "CustomFolder",
            # "output_path": "/custom/path/"
        }
        
        # Run analysis
        result = await analysis_graph.ainvoke(input_data)
        
        print(f"Analysis complete!")
        print(f"Report saved to: {result['pdf_file_path']}")
        if result.get('analysis_summary'):
            print(f"Summary: {result['analysis_summary'][:200]}...")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())