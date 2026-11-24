import type { CSSProperties } from 'react'
import type { ExportedModel, ModelName, Prediction } from '../model'
import { FACTOR_LABELS, fmt, signed } from '../model'

export interface LogEntry {
  id: number
  text: string
  delta: number
}

export function ResultCard(props: {
  prediction: Prediction
  model: ExportedModel
  models: ModelName[]
  onModel: (m: ModelName) => void
}) {
  const { prediction: p, model } = props
  return (
    <section className="hud card" aria-labelledby="result-title">
      <div className="card-head">
        <h2 id="result-title" className="eyebrow">
          Kết quả mô hình {model.name}
        </h2>
        <div className="seg-tabs" role="group" aria-label="Chọn mô hình">
          {props.models.map((m) => (
            <button key={m} type="button" className="tab mono" aria-pressed={m === model.name} onClick={() => props.onModel(m)}>
              {m}
            </button>
          ))}
        </div>
      </div>
      <dl className="stats">
        <div className="stat">
          <dt>Trung bình mọi xe như vậy (95%)</dt>
          <dd>
            {fmt(p.ci[0], 1)} – {fmt(p.ci[1], 1)}
          </dd>
        </div>
        <div className="stat">
          <dt>Một chiếc cụ thể (95%)</dt>
          <dd>
            {fmt(p.pi[0], 1)} – {fmt(p.pi[1], 1)}
          </dd>
        </div>
        <div className="stat">
          <dt>Thải nhiều hơn</dt>
          <dd>{Math.round(p.percentile * 100)}% đội xe 2014</dd>
        </div>
      </dl>
      {p.extrapolating && (
        <p className="warn" role="status">
          Cấu hình nằm xa dữ liệu huấn luyện (đòn bẩy {fmt(p.leverage, 3)} &gt; {fmt(model.leverage_limit, 3)}): con số là ngoại
          suy.
        </p>
      )}
      <p className="model-line mono">
        {model.description} · R² test {fmt(model.metrics.R2, 3)} · RMSE test {fmt(model.metrics.RMSE, 1)} · RMSE CV{' '}
        {fmt(model.metrics.CV_RMSE_mean, 1)} ± {fmt(model.metrics.CV_RMSE_std, 1)} g/km
      </p>
    </section>
  )
}

export function WhyCard({ prediction }: { prediction: Prediction }) {
  const rows = [...prediction.contributions].sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
  const max = Math.max(...rows.map((c) => Math.abs(c.value)), 1)
  return (
    <section className="hud card" aria-labelledby="why-title">
      <h2 id="why-title" className="eyebrow">
        Vì sao ra con số này
      </h2>
      <ul className="why">
        <li>
          <span>Hệ số chặn</span>
          <span className="why-track" aria-hidden="true" />
          <span className="mono">{fmt(prediction.intercept, 1)}</span>
        </li>
        {rows.map((c) => (
          <li key={c.factor}>
            <span>{FACTOR_LABELS[c.factor]}</span>
            <span className="why-track" aria-hidden="true">
              <span className={`why-fill ${c.value < 0 ? 'neg' : 'pos'}`} style={{ '--w': `${(Math.abs(c.value) / max) * 50}%` } as CSSProperties} />
            </span>
            <span className={`mono ${c.value < 0 ? 'down' : c.value > 0 ? 'up' : ''}`}>{signed(c.value)}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}

export function LogCard({ entries }: { entries: LogEntry[] }) {
  return (
    <section className="hud card" aria-labelledby="log-title">
      <h2 id="log-title" className="eyebrow">
        Nhật ký thay đổi
      </h2>
      {entries.length === 0 ? (
        <p className="muted">Chỉnh một thông số: mô hình 3D đổi theo, camera tới đúng bộ phận, và dòng nhật ký ghi lại CO₂ tăng hay giảm bao nhiêu.</p>
      ) : (
        <ol className="log">
          {entries.map((e) => (
            <li key={e.id}>
              <span>{e.text}</span>
              <span className={`mono ${e.delta > 0 ? 'up' : 'down'}`}>{signed(e.delta)} g/km</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
