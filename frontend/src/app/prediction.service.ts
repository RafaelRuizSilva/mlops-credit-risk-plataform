import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

// Espelha o PredictionRequest da API (snake_case preservado de propósito:
// o contrato é da API, o front se adapta a ele)
export interface PredictionRequest {
  revolving_utilization: number;
  age: number;
  n_late_30_59: number;
  debt_ratio: number;
  monthly_income: number | null;
  n_open_credit_lines: number;
  n_late_90: number;
  n_real_estate_loans: number;
  n_late_60_89: number;
  n_dependents: number | null;
}

export interface PredictionResponse {
  prediction_id: string;
  probability_default: number;
  risk_band: 'baixo' | 'medio' | 'alto';
  model_version: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private readonly http = inject(HttpClient);

  // caminho relativo: em dev o proxy.conf redireciona; em prod o nginx faz proxy
  score(request: PredictionRequest): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>('/api/predictions', request);
  }
}
