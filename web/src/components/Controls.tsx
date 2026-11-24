import type { CSSProperties, ReactNode } from 'react'
import type { Body, Config, Fuel, ModelBundle, Trans } from '../model'
import { fmt } from '../model'
import { FUEL_COLOR } from '../scene/Studio'

interface Props {
  config: Config
  bundle: ModelBundle
  thumbs: Partial<Record<Body, string>>
  onChange: <K extends keyof Config>(key: K, value: Config[K]) => void
  onPreset: (preset: Config) => void
}

const BODIES: Body[] = ['small', 'midlarge', 'suv', 'pickup', 'van']
const BODY_SHORT: Record<Body, string> = { small: 'Xe nhỏ', midlarge: 'Sedan', suv: 'SUV', pickup: 'Bán tải', van: 'Van' }
const FUELS: Fuel[] = ['X', 'Z', 'D', 'E']
const TRANS: Trans[] = ['A', 'AS', 'AM', 'AV', 'M']
const CYLINDERS = [3, 4, 5, 6, 8, 10, 12]
const GEARS = [0, 4, 5, 6, 7, 8, 9]

const PRESETS: { label: string; config: Config }[] = [
  { label: 'Xe nhỏ 2.0L', config: { BODY: 'small', ENGINESIZE: 2.0, CYLINDERS: 4, FUELTYPE: 'X', TRANS_TYPE: 'AS', GEARS: 6, HYBRID: 0 } },
  { label: 'SUV 3.5L', config: { BODY: 'suv', ENGINESIZE: 3.5, CYLINDERS: 6, FUELTYPE: 'X', TRANS_TYPE: 'AS', GEARS: 6, HYBRID: 0 } },
  { label: 'SUV 3.5L E85', config: { BODY: 'suv', ENGINESIZE: 3.5, CYLINDERS: 6, FUELTYPE: 'E', TRANS_TYPE: 'AS', GEARS: 6, HYBRID: 0 } },
  { label: 'Sedan hybrid', config: { BODY: 'midlarge', ENGINESIZE: 2.5, CYLINDERS: 4, FUELTYPE: 'X', TRANS_TYPE: 'AV', GEARS: 0, HYBRID: 1 } },
  { label: 'Bán tải V8 5.3L', config: { BODY: 'pickup', ENGINESIZE: 5.3, CYLINDERS: 8, FUELTYPE: 'X', TRANS_TYPE: 'A', GEARS: 6, HYBRID: 0 } },
]

function Group(props: { label: string; hint: string; children: ReactNode }) {
  return (
    <fieldset className="group">
      <legend className="group-label">
        <span>{props.label}</span>
        {props.hint && <span className="group-hint mono">{props.hint}</span>}
      </legend>
      {props.children}
    </fieldset>
  )
}

export function Controls({ config, bundle, thumbs, onChange, onPreset }: Props) {
  const { labels, ranges } = bundle
  const [lo, hi] = ranges.ENGINESIZE
  return (
    <section className="hud panel" aria-labelledby="controls-title">
      <h2 id="controls-title" className="eyebrow">
        Thông số thiết kế
      </h2>

      <Group label="Nhóm thân xe" hint="hình dáng · cỡ bánh · gầm">
        <div className="bodies">
          {BODIES.map((b) => (
            <button key={b} type="button" className="body-btn" aria-pressed={config.BODY === b} title={labels.BODY[b]} onClick={() => onChange('BODY', b)}>
              {thumbs[b] ? <img src={thumbs[b]} alt="" /> : <span className="body-ph" aria-hidden="true" />}
              <span className="mono">{BODY_SHORT[b]}</span>
            </button>
          ))}
        </div>
      </Group>

      <div className="group">
        <label className="group-label" htmlFor="engine">
          <span>Dung tích động cơ</span>
          <span className="group-value">{fmt(config.ENGINESIZE, 1)} L</span>
        </label>
        <span className="group-hint mono">khối máy · ống xả</span>
        <input
          id="engine"
          type="range"
          min={lo}
          max={hi}
          step={0.1}
          value={config.ENGINESIZE}
          style={{ '--fill': `${((config.ENGINESIZE - lo) / (hi - lo)) * 100}%` } as CSSProperties}
          onChange={(e) => onChange('ENGINESIZE', Number(e.target.value))}
        />
        <div className="ticks mono" aria-hidden="true">
          <span>{fmt(lo, 1)}</span>
          <span>{fmt((lo + hi) / 2, 1)}</span>
          <span>{fmt(hi, 1)}</span>
        </div>
      </div>

      <Group label="Số xi-lanh" hint="thẳng hàng ≤ 5, chữ V ≥ 6">
        <div className="grid-btns" style={{ '--n': CYLINDERS.length } as CSSProperties}>
          {CYLINDERS.map((n) => (
            <button key={n} type="button" className="opt mono" aria-pressed={config.CYLINDERS === n} onClick={() => onChange('CYLINDERS', n)}>
              {n}
            </button>
          ))}
        </div>
      </Group>

      <Group label="Nhiên liệu" hint="màu dải đèn, piston">
        <div className="fuels">
          {FUELS.map((f) => (
            <button
              key={f}
              type="button"
              className="opt fuel"
              aria-pressed={config.FUELTYPE === f}
              style={{ '--dot': FUEL_COLOR[f] } as CSSProperties}
              onClick={() => onChange('FUELTYPE', f)}
            >
              <span className="dot" aria-hidden="true" />
              <span className="fuel-name">{labels.FUELTYPE[f]}</span>
              <span className="mono fuel-code">{f}</span>
            </button>
          ))}
        </div>
      </Group>

      <Group label="Kiểu hộp số" hint="bánh răng · puli CVT · cần số sàn">
        <div className="grid-btns" style={{ '--n': TRANS.length } as CSSProperties}>
          {TRANS.map((t) => (
            <button key={t} type="button" className="opt mono" aria-pressed={config.TRANS_TYPE === t} title={labels.TRANS_TYPE[t]} onClick={() => onChange('TRANS_TYPE', t)}>
              {t}
            </button>
          ))}
        </div>
        <div className="grid-btns gears" role="group" aria-label="Số cấp">
          {GEARS.map((g) => (
            <button key={g} type="button" className="opt mono" aria-pressed={config.GEARS === g} onClick={() => onChange('GEARS', g)}>
              {g === 0 ? 'Không ghi' : `${g} cấp`}
            </button>
          ))}
        </div>
      </Group>

      <div className="group hybrid-row">
        <button
          type="button"
          role="switch"
          className="square-switch"
          aria-checked={config.HYBRID === 1}
          aria-labelledby="hybrid-label"
          onClick={() => onChange('HYBRID', config.HYBRID ? 0 : 1)}
        >
          <span aria-hidden="true" />
        </button>
        <span id="hybrid-label" className="group-label">
          <span>Hybrid</span>
          <span className="group-hint mono">pin dưới sàn + mô-tơ</span>
        </span>
      </div>

      <Group label="Thử nhanh" hint="">
        <div className="presets">
          {PRESETS.map((p) => (
            <button key={p.label} type="button" className="opt mono" onClick={() => onPreset(p.config)}>
              {p.label}
            </button>
          ))}
        </div>
      </Group>
    </section>
  )
}
