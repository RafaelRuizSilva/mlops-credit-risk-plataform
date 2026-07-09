/**
 * Modo demonstração (GitHub Pages): score calculado NO NAVEGADOR com os
 * parâmetros reais do modelo baseline (LogisticRegression v1), exportados
 * do MLflow Registry. Mesma matemática do servidor: imputação por mediana
 * -> padronização -> logit -> sigmoide.
 *
 * A plataforma completa (LightGBM v3, auditoria, monitoramento) roda via
 * Docker Compose — veja o repositório.
 */
import { PredictionRequest, PredictionResponse } from './prediction.service';

// ordem das features = ordem do treino
const MEDIANS = [0.15321068, 52, 0, 0.366829146, 5400, 8, 0, 1, 0, 0, 1];
const MEANS = [0.3381708070245003, 52.315044292035765, 0.2810356752972941,
  354.63409006178233, 6410.1086759056325, 8.463720531004425, 0.12585104875873965,
  1.019616830140251, 0.10101750847923732, 0.736197801648347, 0.8020983508195901];
const SCALES = [0.5350113053260874, 14.758634356376962, 1.087916726474189,
  2106.9740543842186, 10951.646213592938, 5.152925026920843, 0.9704652632649658,
  1.1355069035375707, 0.9056778585429726, 1.1076337558865341, 0.39841760306503005];
const COEFS = [0.9174797409065593, -0.3021043335150439, 0.5870695858471789,
  -0.1182653977399694, -0.1886871928564902, 0.13841195987007024, 0.7734454998012622,
  0.1279448335994242, 0.5294525174880796, 0.029289767685787553, 0.012469830841603786];
const INTERCEPT = -0.565085977224742;

export function isDemoMode(): boolean {
  return location.hostname.endsWith('github.io') || location.search.includes('demo=1');
}

export function scoreLocally(req: PredictionRequest): PredictionResponse {
  const raw = [
    req.revolving_utilization, req.age, req.n_late_30_59, req.debt_ratio,
    req.monthly_income, req.n_open_credit_lines, req.n_late_90,
    req.n_real_estate_loans, req.n_late_60_89, req.n_dependents,
    req.monthly_income == null ? 0 : 1,          // declarou_renda (antes da imputação!)
  ];
  let z = INTERCEPT;
  raw.forEach((v, i) => {
    const x = v == null ? MEDIANS[i] : v;        // imputação por mediana do treino
    z += COEFS[i] * ((x - MEANS[i]) / SCALES[i]); // padronização + coeficiente
  });
  const probability = 1 / (1 + Math.exp(-z));
  return {
    prediction_id: crypto.randomUUID(),
    probability_default: probability,
    risk_band: probability < 0.1 ? 'baixo' : probability < 0.3 ? 'medio' : 'alto',
    model_version: '1 (demo no navegador)',
  };
}
