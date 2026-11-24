import type { CSSProperties } from 'react'
import type { ExportedModel, Prediction } from '../model'
import { FACTOR_LABELS, fmt, signed } from '../model'

export interface LogEntry {
  id: number
  text: string
  delta: number
}

export function Readout({ prediction, model }: { prediction: Prediction; model: ExportedModel }) {
  const below = Math.round(prediction.percentile * 100)
  return (
    <section className="panel-block readout" aria-labelledby="readout-title" data-enter>
      <h2 id="readout-title" className="eyebrow">
        CO₂ dự đoán · {model.name}
      </h2>
      <p className="readout-value mono" aria-live="polite">
        {fmt(prediction.mean)}
        <span className="readout-unit">g/km</span>
      </p>
      <dl className="readout-grid">
        <div>
          <dt>Khoảng tin cậy 95%</dt>
          <dd className="mono">
            {fmt(prediction.ci[0])}–{fmt(prediction.ci[1])}
          </dd>
        </div>
        <div>
          <dt>Khoảng dự đoán 95%</dt>
          <dd className="mono">
            {fmt(prediction.pi[0])}–{fmt(prediction.pi[1])}
          </dd>
        </div>
        <div>
          <dt>Thải nhiều hơn</dt>
          <dd className="mono">{below}% đội xe 2014</dd>
        </div>
      </dl>
      {prediction.extrapolating && (
        <p className="warning" role="status">
          Cấu hình này nằm xa dữ liệu huấn luyện (đòn bẩy {fmt(prediction.leverage, 3)} &gt; {fmt(model.leverage_limit, 3)}). Con
          số là ngoại suy, nên đọc thận trọng.
        </p>
      )}
    </section>
  )
}

export function Contributions({ prediction }: { prediction: Prediction }) {
  const max = Math.max(...prediction.contributions.map((c) => Math.abs(c.value)), 1)
  return (
    <section className="panel-block" aria-labelledby="contrib-title">
      <h2 id="contrib-title">Đóng góp từng yếu tố</h2>
      <p className="muted small">
        Hệ số chặn {fmt(prediction.intercept, 1)} g/km cộng các phần dưới đây ra con số dự đoán. Mô hình cộng tính nên mỗi
        phần đọc được riêng.
      </p>
      <ul className="bars">
        {prediction.contributions.map((c) => (
          <li key={c.factor} className="bar-row">
            <span className="bar-label">{FACTOR_LABELS[c.factor]}</span>
            <span className="bar-track" aria-hidden="true">
              <span
                className={`bar-fill ${c.value < 0 ? 'is-neg' : ''}`}
                style={{ '--w': `${(Math.abs(c.value) / max) * 50}%` } as CSSProperties}
              />
            </span>
            <span className="bar-value mono">{signed(c.value)}</span>
          </li>
        ))}
      </ul>
    </section>
  )
}

export function ChangeLog({ entries }: { entries: LogEntry[] }) {
  return (
    <section className="panel-block" aria-labelledby="log-title">
      <h2 id="log-title">Nhật ký thay đổi</h2>
      {entries.length === 0 ? (
        <p className="muted small">Đổi một thông số để xem CO₂ tăng hay giảm bao nhiêu gam.</p>
      ) : (
        <ol className="log">
          {entries.map((e) => (
            <li key={e.id}>
              <span>{e.text}</span>
              <span className={`mono ${e.delta > 0 ? 'is-up' : 'is-down'}`}>{signed(e.delta)} g/km</span>
            </li>
          ))}
        </ol>
      )}
    </section>
  )
}
