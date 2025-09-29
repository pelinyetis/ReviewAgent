import logging
import os
from typing import Optional, Dict, Any
import json
import asyncio
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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

# Safe filename helper
def sanitize_filename(name: str) -> str:
    name = os.path.basename(str(name)).strip()
    # Allow letters, numbers, dot, dash, underscore; replace others with underscore
    name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", name)
    name = name.replace(" ", "_")
    if not name or name in {".", ".."}:
        name = "document"
    # Avoid hidden files
    if name.startswith('.'):
        name = name.lstrip('.') or "document"
    return name

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
    raw_document_name = state["document_name"]
    safe_document_name = sanitize_filename(raw_document_name)

    graph_input = GraphInput(
        document_name=safe_document_name,
        bucket_name=state.get("bucket_name"),
        folder_path=state.get("folder_path"),
        output_path=state.get("output_path"),
    )
    
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
    
    # Validate environment for LLM usage
    if not ANTHROPIC_API_KEY:
        return Command(update={"report_text": "ANTHROPIC_API_KEY bulunamadı. Lütfen ortam değişkenini ayarlayın ve tekrar deneyin."})

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

        # Tamamen Türkçe sistem talimatı (MCP araç adları veya uçlar hardcode edilmeden)
        system_prompt = (
            "Kıdemli bir Türk sözleşme hukukçususun. Bu oturumda MCP aracılığıyla hukuki araştırma "
            "araçlarına erişimin var. İhtiyaç duyduğunda uygun araçları kendi inisiyatifinle çağır, "
            "yanıtlarını özetleyip analizine entegre et. Talimatlar:\n"
            "- Sözleşme türünü, tarafları ve kritik maddeleri tespit et.\n"
            "- Uygulanabilir mevzuatı ve ilgili madde numaralarını tespit etmek için hedefli araştırma yap.\n"
            "- Her atıfta, gerçekten incelediğin madde/metne dayandır ve kaynakları belirt.\n"
            "- Nicel değerleri (tutarlar, tarihler, oranlar) kalın yaz: ör. **10.000 TL**, **%15**, **01/01/2025**.\n"
            "- Araç bir hata verirse veya uygun araç yoksa, bu sınırlamayı belirterek en iyi hukuki değerlendirmeyi yap.\n"
            "Çıktı biçimi: Akıcı Türkçe paragraflar, somut riskler ve uygulanabilir öneriler; sonda önceliklendirilmiş madde işaretli öneriler listesi."
        )

        user_prompt = (
            f"Aşağıdaki sözleşmeyi analiz et. Gerek gördüğünde MCP üzerinden mevzuat ve içtihat araştırması yap.\n\n"
            f"Çıkarılmış değerler (JSON):\n{extracted_json[:3000]}\n\n"
            f"Sözleşme metni (model sınırı için kısaltılmış):\n{contract_text[:MAX_CONTRACT_LENGTH]}\n\n"
            "İstekler:\n"
            "- Sözleşme türünü ve uygulanacak mevzuatı belirle.\n"
            "- Her kilit meselede madde numarasıyla birlikte somut mevzuat atıfları yap.\n"
            "- Tüm nicel değerleri kalın göster.\n"
            "- Türk hukuku ile uyumlu somut riskler ve uygulanabilir öneriler ver."
        )

        # Let LLM freely use tools
        llm_with_tools = llm.bind_tools(mcp_tools)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        iteration = 0
        report_text = ""
        report_parts: list[str] = []
        last_tool_signatures = None
        no_progress_counter = 0
        
        while iteration < MAX_TOOL_ITERATIONS:
            try:
                response = await llm_with_tools.ainvoke(messages)
                messages.append(response)
                
                # Accumulate any narrative content even when tools are called
                if getattr(response, "content", None):
                    content_text = response.content if isinstance(response.content, str) else str(response.content)
                    if content_text.strip():
                        report_parts.append(content_text)

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
                
                # Simple no-progress detection based on repeated tool call signatures
                current_signatures = []
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        try:
                            sig = (tool_call.get("name"), json.dumps(tool_call.get("args", {}), ensure_ascii=False, sort_keys=True)[:200])
                        except Exception:
                            sig = (tool_call.get("name"), "")
                        current_signatures.append(sig)

                if current_signatures == last_tool_signatures:
                    no_progress_counter += 1
                else:
                    no_progress_counter = 0
                last_tool_signatures = current_signatures

                if no_progress_counter >= 1:
                    logger.info("Breaking out due to repeated identical tool calls (no progress)")
                    break

                iteration += 1
                
            except Exception as e:
                logger.error(f"Iteration error: {e}")
                break
        
        # If no report yet, request final report (Türkçe talimat)
        if not report_text:
            final_prompt = (
                "Şimdi bulgularını birleştir ve nihai Türkçe hukuki analiz raporunu yaz. "
                "Çıkarılan değerleri ve araç çıktılarından elde edilen içgörüleri entegre et. "
                "Tüm nicel değerleri kalın göster. Dayandığın her mevzuat/maddeyi belirt. "
                "Sonda önceliklendirilmiş, uygulanabilir öneriler listesi ver."
            )
            final_response = await llm.ainvoke(messages + [HumanMessage(content=final_prompt)])
            base_narrative = "\n\n".join([p for p in report_parts if p.strip()])
            report_text = (base_narrative + ("\n\n" if base_narrative else "") + (final_response.content or "")).strip()
        
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
        
        # English prompts (output remains Turkish)
        system_prompt = """You are a senior Turkish contract lawyer. You do not have access to external legal databases in this session.

Objectives:
- Classify the contract and analyze it under Turkish law using general knowledge.
- Provide cautious, reasonable legal analysis; avoid unverifiable claims.
- Bold all extracted quantitative values (amounts, dates, rates).

Method:
1) Identify parties, consideration, performance, and critical clauses.
2) Map issues to the most likely applicable codes (TBK, TTK, etc.).
3) Discuss legal implications, typical standards, and market/common practice.
4) Flag risks, ambiguities, compliance issues, and negotiation points.
5) Provide concrete, practical recommendations and drafting suggestions.

Output:
- Fluent Turkish paragraphs.
- Bold quantitative values.
- Add a brief disclaimer that live legislation lookups were not performed."""

        user_prompt = f"""Analyze the following contract. No external legal databases are available in this session.

Extracted values (JSON):
{extracted_json[:3000]}

Contract text (truncated to fit model limits):
{contract_text[:MAX_CONTRACT_LENGTH]}

Requirements:
- Determine the contract type and governing laws.
- Bold all extracted quantitative values.
- Provide cautious but concrete risks and actionable recommendations."""

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