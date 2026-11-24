import { useEffect, useRef, useState } from 'react'
import type { Body, Config, Prediction } from '../model'
import { Studio, type Part, type View } from '../scene/Studio'

export interface Change {
  id: number
  part: Part
  delta: number
}

export interface ViewRequest {
  id: number
  view: View
}

interface Props {
  config: Config
  prediction: Prediction
  fleet: number[]
  change: Change | null
  mode: 'single' | 'lineup'
  xray: boolean
  pixelSize: number
  follow: boolean
  autoRotate: boolean
  view: ViewRequest | null
  lineup: { configs: Config[]; co2: number[] }
  onSelectBody: (body: Body) => void
  onThumbnails: (thumbs: Partial<Record<Body, string>>) => void
}

export function Viewport(props: Props) {
  const host = useRef<HTMLDivElement>(null)
  const studio = useRef<Studio | null>(null)
  const lastChange = useRef(0)
  const lastView = useRef(0)
  const [state, setState] = useState<'loading' | 'ready' | 'nogl' | 'error'>(() => (Studio.supported() ? 'loading' : 'nogl'))
  const callbacks = useRef({ select: props.onSelectBody, thumbs: props.onThumbnails })
  callbacks.current = { select: props.onSelectBody, thumbs: props.onThumbnails }

  useEffect(() => {
    if (!host.current || state === 'nogl') return
    let cancelled = false
    let instance: Studio
    try {
      instance = new Studio(host.current, { onSelectBody: (b) => callbacks.current.select(b) })
    } catch {
      setState('nogl')
      return
    }
    instance.setFleet(props.fleet)
    instance
      .load(import.meta.env.BASE_URL)
      .then(() => {
        if (cancelled) return
        studio.current = instance
        callbacks.current.thumbs(instance.thumbnails())
        setState('ready')
      })
      .catch(() => !cancelled && setState('error'))
    return () => {
      cancelled = true
      studio.current = null
      instance.dispose()
    }
    // The studio lives for the lifetime of the component.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    const s = studio.current
    if (!s) return
    const change = props.change && props.change.id !== lastChange.current ? props.change : undefined
    if (change) lastChange.current = change.id
    s.update(props.config, props.prediction.mean, change)
    s.setPiBand(props.prediction.pi[0], props.prediction.pi[1])
    s.updateLineup(props.lineup.configs, props.lineup.co2, props.config.BODY)
  }, [state, props.config, props.prediction, props.change, props.lineup])

  useEffect(() => {
    if (studio.current && props.view && props.view.id !== lastView.current) {
      lastView.current = props.view.id
      studio.current.setView(props.view.view)
    }
  }, [state, props.view])
  useEffect(() => studio.current?.setMode(props.mode), [state, props.mode])
  useEffect(() => studio.current?.setXray(props.xray), [state, props.xray])
  useEffect(() => studio.current?.setPixelSize(props.pixelSize), [state, props.pixelSize])
  useEffect(() => studio.current?.setFollow(props.follow), [state, props.follow])
  useEffect(() => studio.current?.setAutoRotate(props.autoRotate), [state, props.autoRotate])

  return (
    <div className="viewport" role="img" aria-label="Xưởng xe 3D: mô hình xe thay đổi theo thông số">
      <div ref={host} className="viewport-canvas" />
      {state === 'loading' && <p className="viewport-note mono">Đang nạp mẫu xe…</p>}
      {state === 'error' && <p className="viewport-note">Không nạp được mẫu xe. Tải lại trang để thử lại.</p>}
      {state === 'nogl' && (
        <p className="viewport-note">
          Trình duyệt này không bật WebGL nên không vẽ được xưởng 3D. Mọi con số vẫn tính đúng; bật tăng tốc phần cứng để xem
          mô hình xe.
        </p>
      )}
    </div>
  )
}
