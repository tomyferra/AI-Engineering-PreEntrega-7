from app.orchestrator.agents.supervisor import (
    ANALYZE_AVAILABLE_INSTRUCTION,
    FINISH,
    RECOMPUTE_INSTRUCTION,
    SupervisorDecision,
    apply_guards,
    build_supervisor_messages,
)
from app.config import MAX_CALLS_PER_AGENT, MAX_DELEGATIONS
from app.orchestrator.state import ANALYST, RESEARCHER, SYNTHESIZER, Contribution, initial_state


def contribution(agent: str, attempt: int = 1, issues=()) -> Contribution:
    return Contribution(agent=agent, attempt=attempt, instruction="i", content="dato",
                        tool_calls=[{"tool": "calcular", "args": {"expresion": "1 + 1"}, "result": "2"}],
                        issues=list(issues))


def state_with(contributions=(), delegations=0) -> dict:
    return {**initial_state("consulta"), "contributions": list(contributions), "delegations": delegations}


def test_finalizar_mapea_al_sintetizador_sin_instruccion():
    routing = apply_guards(SupervisorDecision(reasoning="ok", next=FINISH, instruction="x"), state_with())
    assert routing["next"] == SYNTHESIZER and routing["instruction"] == "" and not routing["forced"]


def test_respeta_la_decision_del_llm_dentro_de_los_limites():
    routing = apply_guards(SupervisorDecision(reasoning="r", next=ANALYST, instruction="calcular"),
                           state_with([contribution(RESEARCHER)], delegations=1))
    assert routing == {"step": 1, "next": ANALYST, "instruction": "calcular", "reasoning": "r", "forced": False}


def test_limite_global_de_delegaciones_fuerza_la_sintesis():
    routing = apply_guards(SupervisorDecision(reasoning="r", next=RESEARCHER, instruction="más"),
                           state_with([contribution(RESEARCHER)], delegations=MAX_DELEGATIONS))
    assert routing["next"] == SYNTHESIZER and routing["forced"]


def test_limite_del_investigador_sin_analisis_deriva_al_analista():
    previos = [contribution(RESEARCHER, i + 1) for i in range(MAX_CALLS_PER_AGENT)]
    routing = apply_guards(SupervisorDecision(reasoning="r", next=RESEARCHER, instruction="más"),
                           state_with(previos, delegations=MAX_CALLS_PER_AGENT))
    assert routing["next"] == ANALYST and routing["forced"]
    assert routing["instruction"] == ANALYZE_AVAILABLE_INSTRUCTION


def test_limite_por_agente_fuerza_la_sintesis():
    previos = [contribution(RESEARCHER, i + 1) for i in range(MAX_CALLS_PER_AGENT)] + [contribution(ANALYST)]
    routing = apply_guards(SupervisorDecision(reasoning="r", next=RESEARCHER, instruction="más"),
                           state_with(previos, delegations=MAX_CALLS_PER_AGENT))
    assert routing["next"] == SYNTHESIZER and routing["forced"]


def test_analista_sin_datos_redirige_al_investigador():
    routing = apply_guards(SupervisorDecision(reasoning="r", next=ANALYST, instruction="calcular"), state_with())
    assert routing["next"] == RESEARCHER and routing["forced"]


def test_supervisor_ve_aportes_y_presupuesto_pero_no_el_historial_crudo():
    state = state_with([contribution(RESEARCHER)], delegations=2)
    _system, human = build_supervisor_messages(state)
    assert "Aporte de investigador" in human.text
    assert 'calcular({"expresion": "1 + 1"}) -> 2' in human.text  # evidencia real para validar con la rúbrica
    assert f"Delegaciones restantes: {MAX_DELEGATIONS - 2}" in human.text


def test_finalizar_con_analisis_desactualizado_fuerza_recalculo():
    aportes = [contribution(RESEARCHER), contribution(ANALYST), contribution(RESEARCHER, 2)]
    routing = apply_guards(SupervisorDecision(reasoning="ok", next=FINISH), state_with(aportes, delegations=3))
    assert routing["next"] == ANALYST and routing["forced"]
    assert routing["instruction"] == RECOMPUTE_INSTRUCTION


def test_analisis_desactualizado_sin_presupuesto_finaliza_igual():
    aportes = [contribution(RESEARCHER), contribution(ANALYST), contribution(RESEARCHER, 2)]
    routing = apply_guards(SupervisorDecision(reasoning="ok", next=FINISH),
                           state_with(aportes, delegations=MAX_DELEGATIONS))
    assert routing["next"] == SYNTHESIZER and not routing["forced"]


def test_aporte_no_validado_fuerza_correccion_del_investigador():
    aportes = [contribution(RESEARCHER, issues=["El monto 36.500.000 no aparece en ninguna fuente."])]
    routing = apply_guards(SupervisorDecision(reasoning="ok", next=ANALYST, instruction="calcular"),
                           state_with(aportes, delegations=1))
    assert routing["next"] == RESEARCHER and routing["forced"]
    assert "36.500.000" in routing["instruction"]


def test_aporte_no_validado_sin_presupuesto_no_bloquea():
    aportes = [contribution(RESEARCHER, i + 1, issues=["x"]) for i in range(MAX_CALLS_PER_AGENT)]
    routing = apply_guards(SupervisorDecision(reasoning="ok", next=ANALYST, instruction="calcular"),
                           state_with(aportes, delegations=MAX_CALLS_PER_AGENT))
    assert routing["next"] == ANALYST and not routing["forced"]
