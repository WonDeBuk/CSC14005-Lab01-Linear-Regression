import type { CSSProperties } from 'react'
import type { Body, Config, Fuel, ModelBundle, Trans } from '../model'
import { fmt } from '../model'

interface Props {
  config: Config
  bundle: ModelBundle
  onChange: <K extends keyof Config>(key: K, value: Config[K]) => void
  onReset: () => void
}

const BODIES: Body[] = ['small', 'midlarge', 'suv', 'pickup', 'van']
const FUELS: Fuel[] = ['X', 'Z', 'D', 'E']
const TRANS: Trans[] = ['A', 'AS', 'AM', 'AV', 'M']
const BODY_SHORT: Record<Body, string> = { small: 'Xe nhỏ', midlarge: 'Sedan', suv: 'SUV', pickup: 'Bán tải', van: 'Van' }

function Segmented<T extends string>(props: {
  legend: string
  options: T[]
  value: T
  label: (v: T) => string
  short?: (v: T) => string
  onPick: (v: T) => void
}) {
  return (
    <fieldset className="field">
      <legend className="field-label">{props.legend}</legend>
      <div className="segmented" style={{ '--n': props.options.length } as CSSProperties}>
        {props.options.map((o) => (
          <button
            key={o}
            type="button"
            className="seg"
            aria-pressed={o === props.value}
            title={props.label(o)}
            onClick={() => props.onPick(o)}
          >
            {props.short ? props.short(o) : props.label(o)}
          </button>
        ))}
      </div>
    </fieldset>
  )
}

function Slider(props: {
  id: string
  label: string
  value: number
  min: number
  max: number
  step: number
  display: string
  onInput: (v: number) => void
}) {
  return (
    <div className="field">
      <div className="field-row">
        <label className="field-label" htmlFor={props.id}>
          {props.label}
        </label>
        <output className="mono" htmlFor={props.id}>
          {props.display}
        </output>
      </div>
      <input
        id={props.id}
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        style={{ '--fill': `${((props.value - props.min) / (props.max - props.min)) * 100}%` } as CSSProperties}
        onChange={(e) => props.onInput(Number(e.target.value))}
      />
    </div>
  )
}

export function Controls({ config, bundle, onChange, onReset }: Props) {
  const { labels, ranges } = bundle
  const cvt = config.TRANS_TYPE === 'AV'
  return (
    <section className="panel-block" aria-labelledby="controls-title">
      <div className="block-head">
        <h2 id="controls-title">Thông số thiết kế</h2>
        <button type="button" className="btn btn-ghost" onClick={onReset}>
          Đặt lại theo nhóm xe
        </button>
      </div>
      <Segmented
        legend="Nhóm thân xe"
        options={BODIES}
        value={config.BODY}
        label={(v) => labels.BODY[v]}
        short={(v) => BODY_SHORT[v]}
        onPick={(v) => onChange('BODY', v)}
      />
      <Slider
        id="engine"
        label="Dung tích động cơ"
        value={config.ENGINESIZE}
        min={ranges.ENGINESIZE[0]}
        max={ranges.ENGINESIZE[1]}
        step={0.1}
        display={`${fmt(config.ENGINESIZE, 1)} L`}
        onInput={(v) => onChange('ENGINESIZE', v)}
      />
      <Slider
        id="cylinders"
        label="Số xi-lanh"
        value={config.CYLINDERS}
        min={ranges.CYLINDERS[0]}
        max={ranges.CYLINDERS[1]}
        step={1}
        display={`${config.CYLINDERS}`}
        onInput={(v) => onChange('CYLINDERS', v)}
      />
      <Segmented
        legend="Nhiên liệu"
        options={FUELS}
        value={config.FUELTYPE}
        label={(v) => labels.FUELTYPE[v]}
        onPick={(v) => onChange('FUELTYPE', v)}
      />
      <Segmented
        legend="Kiểu hộp số"
        options={TRANS}
        value={config.TRANS_TYPE}
        label={(v) => labels.TRANS_TYPE[v]}
        short={(v) => v}
        onPick={(v) => onChange('TRANS_TYPE', v)}
      />
      <Slider
        id="gears"
        label={cvt ? 'Số cấp (0 = vô cấp, không ghi số cấp)' : 'Số cấp hộp số'}
        value={config.GEARS}
        min={cvt ? 0 : 4}
        max={ranges.GEARS[1]}
        step={1}
        display={config.GEARS === 0 ? 'CVT' : `${config.GEARS} cấp`}
        onInput={(v) => onChange('GEARS', v)}
      />
      <div className="field field-row">
        <span className="field-label" id="hybrid-label">
          Công nghệ hybrid
        </span>
        <button
          type="button"
          role="switch"
          className="switch"
          aria-checked={config.HYBRID === 1}
          aria-labelledby="hybrid-label"
          onClick={() => onChange('HYBRID', config.HYBRID ? 0 : 1)}
        >
          <span className="switch-knob" />
        </button>
      </div>
    </section>
  )
}
