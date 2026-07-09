import { Component, inject, signal } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { PercentPipe } from '@angular/common';

import { isDemoMode } from './demo-model';
import { PredictionRequest, PredictionResponse, PredictionService } from './prediction.service';

@Component({
  selector: 'app-root',
  imports: [ReactiveFormsModule, PercentPipe],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  private readonly fb = inject(FormBuilder);
  private readonly predictions = inject(PredictionService);

  protected readonly result = signal<PredictionResponse | null>(null);
  protected readonly loading = signal(false);
  protected readonly error = signal<string | null>(null);
  protected readonly demo = isDemoMode();

  // validações espelham o schema Pydantic (o 422 da API é a segunda linha de defesa)
  protected readonly form = this.fb.nonNullable.group({
    revolving_utilization: [0.5, [Validators.required, Validators.min(0)]],
    age: [35, [Validators.required, Validators.min(18), Validators.max(120)]],
    n_late_30_59: [0, [Validators.required, Validators.min(0)]],
    debt_ratio: [0.3, [Validators.required, Validators.min(0)]],
    monthly_income: this.fb.control<number | null>(null, [Validators.min(0)]),
    n_open_credit_lines: [5, [Validators.required, Validators.min(0)]],
    n_late_90: [0, [Validators.required, Validators.min(0)]],
    n_real_estate_loans: [0, [Validators.required, Validators.min(0)]],
    n_late_60_89: [0, [Validators.required, Validators.min(0)]],
    n_dependents: this.fb.control<number | null>(null, [Validators.min(0)]),
  });

  protected readonly bandLabel: Record<string, string> = {
    baixo: 'Risco baixo',
    medio: 'Risco médio',
    alto: 'Risco alto',
  };

  protected submit(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }
    this.loading.set(true);
    this.error.set(null);
    this.predictions.score(this.form.getRawValue() as PredictionRequest).subscribe({
      next: (response) => {
        this.result.set(response);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(
          err.status === 503
            ? 'Serviço de scoring indisponível no momento. Tente novamente.'
            : 'Não foi possível avaliar a solicitação. Verifique os campos.'
        );
        this.loading.set(false);
      },
    });
  }

  protected reset(): void {
    this.result.set(null);
    this.error.set(null);
  }
}
