import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, delay, of } from 'rxjs';

import { isDemoMode, scoreLocally } from './demo-model';

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

  score(request: PredictionRequest): Observable<PredictionResponse> {
    // GitHub Pages não tem backend: o score roda no navegador (demo-model.ts)
    if (this.demo) {
      return of(scoreLocally(request)).pipe(delay(350));
    }
    // caminho relativo: em dev o proxy.conf redireciona; em prod o nginx faz proxy
    return this.http.post<PredictionResponse>('/api/predictions', request);
  }

  get demo(): boolean {
    return isDemoMode();
  }
}
