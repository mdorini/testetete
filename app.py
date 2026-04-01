import os
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Gerador de Recurso de Glosa Automático")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """Você é um especialista em direito médico, saúde suplementar e codificação TUSS, com profundo conhecimento das diretrizes da ANS (Agência Nacional de Saúde Suplementar), CONITEC (Comissão Nacional de Incorporação de Tecnologias no SUS), AMB (Associação Médica Brasileira), CFM (Conselho Federal de Medicina) e das principais sociedades de especialidade médica do Brasil.

Sua tarefa é redigir cartas de recurso de glosa formais, técnicas e persuasivas para clínicas e médicos brasileiros.

Ao gerar o recurso, você deve:
1. Usar linguagem jurídico-médica formal em português brasileiro
2. Citar as normas da ANS pertinentes (ex: RN 465/2021, RN 558/2022, Rol de Procedimentos vigente)
3. Referenciar diretrizes clínicas da CONITEC, AMB, CFM e sociedades de especialidade conforme aplicável ao procedimento
4. Argumentar com base na necessidade médica, no código TUSS informado e na documentação clínica fornecida
5. Mencionar o direito do beneficiário à cobertura e o dever da operadora de cobrir procedimentos previstos no Rol
6. Solicitar a revisão da glosa de forma clara e assertiva
7. Estruturar a carta com: cabeçalho, identificação do processo, fundamentação técnica, fundamentação jurídica, pedido e fechamento

A carta deve ser objetiva, convincente e baseada nos dados fornecidos."""


class GlosaRequest(BaseModel):
    codigo_tuss: str
    nome_procedimento: str
    motivo_glosa: str
    justificativa_clinica: str
    nome_paciente: str = ""
    nome_medico: str = ""
    crm_medico: str = ""
    especialidade: str = ""
    nome_operadora: str = ""
    numero_guia: str = ""
    data_procedimento: str = ""
    documentacao_adicional: str = ""


def build_user_prompt(req: GlosaRequest) -> str:
    partes = []
    partes.append(f"**Código TUSS:** {req.codigo_tuss}")
    partes.append(f"**Procedimento:** {req.nome_procedimento}")
    partes.append(f"**Motivo da Glosa:** {req.motivo_glosa}")
    partes.append(f"**Justificativa Clínica:** {req.justificativa_clinica}")

    if req.nome_paciente:
        partes.append(f"**Paciente:** {req.nome_paciente}")
    if req.nome_medico:
        partes.append(f"**Médico:** Dr(a). {req.nome_medico}")
    if req.crm_medico:
        partes.append(f"**CRM:** {req.crm_medico}")
    if req.especialidade:
        partes.append(f"**Especialidade:** {req.especialidade}")
    if req.nome_operadora:
        partes.append(f"**Operadora de Saúde:** {req.nome_operadora}")
    if req.numero_guia:
        partes.append(f"**Número da Guia:** {req.numero_guia}")
    if req.data_procedimento:
        partes.append(f"**Data do Procedimento:** {req.data_procedimento}")
    if req.documentacao_adicional:
        partes.append(f"**Documentação e Informações Adicionais:** {req.documentacao_adicional}")

    dados = "\n".join(partes)

    return f"""Com base nos dados abaixo, redija uma carta de recurso de glosa completa, formal e tecnicamente fundamentada:

{dados}

Gere a carta de recurso completa, incluindo todos os argumentos técnicos, citações de diretrizes (CONITEC, AMB, CFM, sociedades de especialidade) e fundamentação jurídica na ANS. A carta deve estar pronta para ser enviada à operadora."""


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/gerar-recurso")
async def gerar_recurso(req: GlosaRequest):
    def stream_generator():
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(req)}],
        ) as stream:
            for event in stream:
                if (
                    event.type == "content_block_delta"
                    and event.delta.type == "text_delta"
                ):
                    yield event.delta.text

    return StreamingResponse(stream_generator(), media_type="text/plain; charset=utf-8")
