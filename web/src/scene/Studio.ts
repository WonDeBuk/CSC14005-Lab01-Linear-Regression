// The pixelated 3D workshop: one car rig (or a lineup of five) driven by the
// configuration and the model's prediction. Plain three.js, owned by <Viewport>.

import * as THREE from 'three'
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js'
import { RenderPixelatedPass } from 'three/addons/postprocessing/RenderPixelatedPass.js'
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js'
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js'
import type { Body, Config, Fuel } from '../model'
import { fmt, signed } from '../model'

export const BODIES: Body[] = ['small', 'midlarge', 'suv', 'pickup', 'van']
const MODEL_FILE: Record<Body, string> = {
  small: 'hatchback-sports',
  midlarge: 'sedan',
  suv: 'suv',
  pickup: 'truck',
  van: 'van',
}
export const FUEL_COLOR: Record<Fuel, string> = { X: '#0d7480', Z: '#2f6fd6', D: '#3b4a52', E: '#3f9b4f' }

const CAR_LENGTH = 4
const RING_RADIUS = 3.7
const CO2_MIN = 100
const CO2_MAX = 500
const RING_SPAN = Math.PI * 1.7
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches

export type Part = 'body' | 'engine' | 'gearbox' | 'hybrid' | 'fuel' | 'exhaust'

export function smokeColor(co2: number): THREE.Color {
  const stops: [number, string][] = [
    [150, '#7cc7a8'],
    [250, '#e9c46a'],
    [380, '#e07a3f'],
  ]
  if (co2 <= stops[0][0]) return new THREE.Color(stops[0][1])
  for (let i = 1; i < stops.length; i++) {
    if (co2 <= stops[i][0]) {
      const k = (co2 - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0])
      return new THREE.Color(stops[i - 1][1]).lerp(new THREE.Color(stops[i][1]), k)
    }
  }
  return new THREE.Color(stops[stops.length - 1][1])
}

function co2Angle(co2: number) {
  const k = THREE.MathUtils.clamp((co2 - CO2_MIN) / (CO2_MAX - CO2_MIN), 0, 1)
  return Math.PI * 1.35 - k * RING_SPAN // clockwise seen from above, gap at the back
}

const ease = (k: number) => 1 - Math.pow(1 - k, 3)

interface Tween {
  start: number
  duration: number
  update: (k: number) => void
}

function std(color: string, extra: Partial<THREE.MeshStandardMaterialParameters> = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.7, metalness: 0.1, flatShading: true, ...extra })
}

/** One car: Kenney shell plus the parts that show the design inputs. */
class CarRig {
  root = new THREE.Group()
  shell = new THREE.Group()
  internals = new THREE.Group()
  engine = new THREE.Group()
  block: THREE.Mesh
  pistons = new THREE.Group()
  gearbox = new THREE.Group()
  battery: THREE.Mesh
  motor: THREE.Mesh
  strips: THREE.Mesh[] = []
  exhaust = new THREE.Group()
  shellMaterials: THREE.MeshStandardMaterial[] = []
  size = new THREE.Vector3(2, 1.4, CAR_LENGTH)
  body: Body | null = null
  config: Config | null = null

  constructor(withInternals: boolean) {
    this.root.add(this.shell)
    this.block = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), std('#5d6b72', { metalness: 0.4 }))
    this.engine.add(this.block, this.pistons)
    this.battery = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), std('#1f3038', { emissive: '#35d0c5', emissiveIntensity: 0.9 }))
    this.motor = new THREE.Mesh(new THREE.TorusGeometry(0.28, 0.07, 6, 16), std('#1f3038', { emissive: '#35d0c5', emissiveIntensity: 1.2 }))
    for (const side of [-1, 1]) {
      const strip = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.06, 1), std('#0d7480', { emissive: '#0d7480', emissiveIntensity: 0.8 }))
      strip.userData.side = side
      this.strips.push(strip)
      this.root.add(strip)
    }
    this.internals.add(this.engine, this.gearbox, this.battery, this.motor)
    if (withInternals) this.root.add(this.internals)
    this.root.add(this.exhaust)
  }

  setShell(template: THREE.Object3D, body: Body) {
    this.shell.clear()
    this.shellMaterials = []
    const clone = template.clone(true)
    clone.traverse((o) => {
      const mesh = o as THREE.Mesh
      if (!mesh.isMesh) return
      mesh.castShadow = true
      mesh.receiveShadow = true
      const material = (mesh.material as THREE.MeshStandardMaterial).clone()
      material.transparent = true
      mesh.material = material
      this.shellMaterials.push(material)
    })
    this.shell.add(clone)
    this.body = body
    new THREE.Box3().setFromObject(clone).getSize(this.size)
    this.layout()
  }

  /** Place the parts relative to the shell's size. */
  layout() {
    const { x: W, y: H, z: L } = this.size
    this.engine.position.set(0, H * 0.34, L * 0.3)
    this.gearbox.position.set(0, H * 0.22, L * 0.08)
    this.battery.scale.set(W * 0.55, 0.1, L * 0.32)
    this.battery.position.set(0, 0.2, -L * 0.08)
    this.motor.position.set(0, 0.32, -L * 0.3)
    this.motor.rotation.y = Math.PI / 2
    for (const strip of this.strips) {
      strip.scale.z = L * 0.6
      strip.position.set((strip.userData.side as number) * (W / 2 + 0.01), H * 0.26, 0)
    }
    this.exhaust.position.set(0, 0.22, -L / 2 - 0.02)
  }

  apply(c: Config) {
    this.config = c
    const fuel = new THREE.Color(FUEL_COLOR[c.FUELTYPE])

    // Engine block grows with displacement.
    const s = 0.55 + c.ENGINESIZE * 0.085
    this.block.scale.set(0.55 * s, 0.4 * s, 0.7 * s)

    // Pistons: inline up to 5, V above.
    this.pistons.clear()
    const pistonGeo = new THREE.CylinderGeometry(0.055 * s, 0.055 * s, 0.2 * s, 8)
    const pistonMat = std(FUEL_COLOR[c.FUELTYPE], { emissive: fuel, emissiveIntensity: 0.5 })
    const banks = c.CYLINDERS <= 5 ? [[0, 0, c.CYLINDERS]] : [[-1, 0.45, Math.ceil(c.CYLINDERS / 2)], [1, -0.45, Math.floor(c.CYLINDERS / 2)]]
    for (const [side, tilt, count] of banks) {
      for (let i = 0; i < count; i++) {
        const p = new THREE.Mesh(pistonGeo, pistonMat)
        const z = ((i + 0.5) / count - 0.5) * 0.62 * s
        p.position.set(side * 0.13 * s, 0.26 * s, z)
        p.rotation.z = tilt
        this.pistons.add(p)
      }
    }

    // Gearbox: one gear per ratio, CVT as two pulleys and a belt, manual adds a shifter.
    this.gearbox.clear()
    const gearMat = std('#9ea7ab', { metalness: 0.5 })
    if (c.GEARS === 0) {
      for (const [x, r] of [[-0.16, 0.16], [0.16, 0.1]]) {
        const pulley = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.08, 12), gearMat)
        pulley.rotation.z = Math.PI / 2
        pulley.position.set(0, 0, x)
        this.gearbox.add(pulley)
      }
      const belt = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.32, 0.4), std('#1f3038'))
      this.gearbox.add(belt)
    } else {
      for (let i = 0; i < c.GEARS; i++) {
        const r = 0.17 - i * 0.012
        const gear = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 0.05, 8), gearMat)
        gear.rotation.z = Math.PI / 2
        gear.position.set(((i + 0.5) / c.GEARS - 0.5) * 0.7, 0, 0)
        this.gearbox.add(gear)
      }
    }
    if (c.TRANS_TYPE === 'M') {
      const stick = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.45, 6), std('#1f3038'))
      stick.position.set(0, 0.25, -0.1)
      const knob = new THREE.Mesh(new THREE.SphereGeometry(0.06, 8, 6), std('#0d7480'))
      knob.position.set(0, 0.48, -0.1)
      this.gearbox.add(stick, knob)
    }

    this.battery.visible = this.motor.visible = c.HYBRID === 1

    for (const strip of this.strips) {
      const m = strip.material as THREE.MeshStandardMaterial
      m.color.copy(fuel)
      m.emissive.copy(fuel)
    }

    // Exhaust pipes: 1, 2 or 4 by cylinders.
    this.exhaust.clear()
    const pipes = c.CYLINDERS <= 4 ? 1 : c.CYLINDERS <= 8 ? 2 : 4
    const pipeR = 0.04 + c.ENGINESIZE * 0.008
    for (let i = 0; i < pipes; i++) {
      const pipe = new THREE.Mesh(new THREE.CylinderGeometry(pipeR, pipeR, 0.25, 8), std('#5d6b72', { metalness: 0.6 }))
      pipe.rotation.x = Math.PI / 2
      pipe.position.x = pipes === 1 ? this.size.x * 0.25 : ((i + 0.5) / pipes - 0.5) * this.size.x * 0.6
      this.exhaust.add(pipe)
    }
  }

  setOpacity(opacity: number) {
    for (const m of this.shellMaterials) {
      m.opacity = opacity
      m.depthWrite = opacity > 0.99
    }
  }

  partObject(part: Part): THREE.Object3D {
    return { body: this.shell, engine: this.engine, gearbox: this.gearbox, hybrid: this.battery, fuel: this.strips[1], exhaust: this.exhaust }[part]
  }
}

interface Particle {
  alive: boolean
  pos: THREE.Vector3
  vel: THREE.Vector3
  age: number
  life: number
  size: number
}

export interface StudioOptions {
  onSelectBody?: (body: Body) => void
}

export class Studio {
  private renderer: THREE.WebGLRenderer
  private labels: CSS2DRenderer
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100)
  private controls: OrbitControls
  private composer: EffectComposer
  private pixelPass: RenderPixelatedPass
  private pixelSize = 4
  private templates = new Map<Body, THREE.Object3D>()
  private rig = new CarRig(true)
  private lineup: CarRig[] = BODIES.map(() => new CarRig(false))
  private lineupGroup = new THREE.Group()
  private columns: THREE.Mesh[] = []
  private columnLabels: CSS2DObject[] = []
  private ring = new THREE.Group()
  private ringBars!: THREE.InstancedMesh
  private piBand!: THREE.Mesh
  private marker!: THREE.Mesh
  private smoke!: THREE.InstancedMesh
  private particles: Particle[] = []
  private emitCarry = 0
  private co2 = 250
  private tweens: Tween[] = []
  private xrayUntil = 0
  private xrayLocked = false
  private opacity = 1
  private mode: 'single' | 'lineup' = 'single'
  private lastFrame = performance.now()
  private frame = 0
  private resizeObserver: ResizeObserver
  private raycaster = new THREE.Raycaster()
  private container: HTMLElement
  private options: StudioOptions

  static supported(): boolean {
    try {
      const canvas = document.createElement('canvas')
      return !!(canvas.getContext('webgl2') || canvas.getContext('webgl'))
    } catch {
      return false
    }
  }

  constructor(container: HTMLElement, options: StudioOptions = {}) {
    this.container = container
    this.options = options
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' })
    this.renderer.setPixelRatio(1)
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFShadowMap
    container.appendChild(this.renderer.domElement)

    this.labels = new CSS2DRenderer()
    this.labels.domElement.className = 'studio-labels'
    container.appendChild(this.labels.domElement)

    this.scene.background = new THREE.Color('#e3eaed')
    this.camera.position.set(7.5, 4.6, 8.5)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.target.set(0, 0.7, 0)
    this.controls.enableDamping = true
    this.controls.minDistance = 4
    this.controls.maxDistance = 24
    this.controls.maxPolarAngle = Math.PI * 0.47

    this.composer = new EffectComposer(this.renderer)
    this.pixelPass = new RenderPixelatedPass(this.pixelSize, this.scene, this.camera, {
      normalEdgeStrength: 0.35,
      depthEdgeStrength: 0.45,
    })
    this.composer.addPass(this.pixelPass)
    this.composer.addPass(new OutputPass())

    this.buildStage()
    this.buildRing()
    this.buildSmoke()
    this.scene.add(this.rig.root, this.lineupGroup)
    this.buildLineup()
    this.lineupGroup.visible = false

    this.renderer.domElement.addEventListener('pointerdown', this.onPointer)
    this.resizeObserver = new ResizeObserver(() => this.resize())
    this.resizeObserver.observe(container)
    this.resize()
    this.loop()
  }

  async load(baseUrl: string): Promise<void> {
    const loader = new GLTFLoader()
    await Promise.all(
      BODIES.map(async (body) => {
        const gltf = await loader.loadAsync(`${baseUrl}models/${MODEL_FILE[body]}.glb`)
        const model = gltf.scene
        model.traverse((o) => {
          const mat = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined
          if (mat?.map) {
            mat.map.magFilter = THREE.NearestFilter
            mat.map.minFilter = THREE.NearestFilter
          }
        })
        const box = new THREE.Box3().setFromObject(model)
        const size = box.getSize(new THREE.Vector3())
        model.scale.setScalar(CAR_LENGTH / size.z)
        const scaled = new THREE.Box3().setFromObject(model)
        const center = scaled.getCenter(new THREE.Vector3())
        model.position.set(-center.x, -scaled.min.y, -center.z)
        const wrapper = new THREE.Group()
        wrapper.add(model)
        this.templates.set(body, wrapper)
      }),
    )
    BODIES.forEach((body, i) => this.lineup[i].setShell(this.templates.get(body)!, body))
  }

  // ---- public API -------------------------------------------------------

  /** Apply a configuration; `changed` names the part that moved (for the flash, label and camera). */
  update(c: Config, co2: number, changed?: { part: Part; delta: number }) {
    const bodyChanged = this.rig.body !== c.BODY
    if (bodyChanged) {
      this.rig.setShell(this.templates.get(c.BODY)!, c.BODY)
      this.rig.setOpacity(this.opacity)
      if (this.rig.config && !REDUCED) this.squash(this.rig.shell)
    }
    this.rig.apply(c)
    this.co2 = co2
    this.marker.position.copy(this.ringPoint(co2, RING_RADIUS))
    this.marker.position.y = 0.45
    ;(this.marker.material as THREE.MeshStandardMaterial).color.copy(smokeColor(co2))

    if (changed && this.mode === 'single') {
      const target = this.rig.partObject(changed.part)
      this.flash(target)
      this.deltaLabel(target, changed.delta)
      if (['engine', 'gearbox', 'hybrid'].includes(changed.part)) this.xrayUntil = performance.now() + 2600
      this.flyTo(changed.part)
    }
  }

  updateLineup(configs: Config[], co2: number[], selected: Body) {
    configs.forEach((c, i) => {
      const rig = this.lineup[i]
      rig.apply(c)
      const h = Math.max(0.05, (co2[i] - 50) * 0.011)
      const column = this.columns[i]
      column.scale.y = h
      column.position.y = h / 2
      ;(column.material as THREE.MeshStandardMaterial).color.copy(smokeColor(co2[i]))
      const label = this.columnLabels[i]
      label.position.set(column.position.x, h + 0.35, column.position.z)
      label.element.textContent = `${fmt(co2[i])} g/km`
      label.element.classList.toggle('is-selected', BODIES[i] === selected)
      rig.setOpacity(BODIES[i] === selected ? 1 : 0.55)
    })
  }

  setPiBand(lo: number, hi: number) {
    const a0 = co2Angle(hi)
    const a1 = co2Angle(lo)
    this.piBand.geometry.dispose()
    this.piBand.geometry = new THREE.RingGeometry(RING_RADIUS - 0.32, RING_RADIUS + 0.32, 48, 1, a0, a1 - a0)
  }

  setMode(mode: 'single' | 'lineup') {
    this.mode = mode
    this.rig.root.visible = this.ring.visible = this.smoke.visible = mode === 'single'
    this.lineupGroup.visible = mode === 'lineup'
    for (const label of this.columnLabels) label.visible = mode === 'lineup'
    if (mode === 'lineup') this.moveCamera(new THREE.Vector3(0, 10, 21), new THREE.Vector3(0, 1.2, -0.6))
    else this.moveCamera(new THREE.Vector3(7.5, 4.6, 8.5), new THREE.Vector3(0, 0.7, 0))
  }

  setPixelSize(size: number) {
    this.pixelSize = size
    this.pixelPass.setPixelSize(Math.max(1, size))
  }

  setXray(on: boolean) {
    this.xrayLocked = on
  }

  dispose() {
    cancelAnimationFrame(this.frame)
    this.resizeObserver.disconnect()
    this.renderer.domElement.removeEventListener('pointerdown', this.onPointer)
    this.controls.dispose()
    this.renderer.dispose()
    this.container.replaceChildren()
  }

  // ---- scene building ---------------------------------------------------

  private buildStage() {
    const hemi = new THREE.HemisphereLight('#ffffff', '#b9c5ca', 1.6)
    const sun = new THREE.DirectionalLight('#ffffff', 2.4)
    sun.position.set(6, 10, 4)
    sun.castShadow = true
    sun.shadow.mapSize.set(1024, 1024)
    const cam = sun.shadow.camera as THREE.OrthographicCamera
    cam.left = cam.bottom = -12
    cam.right = cam.top = 12
    this.scene.add(hemi, sun)

    const floor = new THREE.Mesh(new THREE.CircleGeometry(30, 48), std('#d7e0e4', { roughness: 1 }))
    floor.rotation.x = -Math.PI / 2
    floor.receiveShadow = true
    const platform = new THREE.Mesh(new THREE.CylinderGeometry(RING_RADIUS + 0.6, RING_RADIUS + 0.8, 0.12, 48), std('#ecf0f2', { roughness: 0.9 }))
    platform.position.y = -0.06
    platform.receiveShadow = true
    const grid = new THREE.GridHelper(40, 40, '#c6d1d6', '#c6d1d6')
    grid.position.y = 0.001
    this.scene.add(floor, grid, platform)
  }

  private ringPoint(co2: number, radius: number) {
    const a = co2Angle(co2)
    return new THREE.Vector3(Math.cos(a) * radius, 0, -Math.sin(a) * radius)
  }

  private buildRing() {
    const bins = 64
    const geo = new THREE.BoxGeometry(0.16, 1, 0.16)
    this.ringBars = new THREE.InstancedMesh(geo, std('#9ea7ab'), bins)
    this.ringBars.castShadow = true
    this.ring.add(this.ringBars)

    this.piBand = new THREE.Mesh(
      new THREE.RingGeometry(1, 1.1, 8),
      new THREE.MeshBasicMaterial({ color: '#0d7480', transparent: true, opacity: 0.28, side: THREE.DoubleSide }),
    )
    this.piBand.rotation.x = -Math.PI / 2
    this.piBand.position.y = 0.012
    this.ring.add(this.piBand)

    this.marker = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.9, 0.22), std('#e9c46a', { emissive: '#0d7480', emissiveIntensity: 0.15 }))
    this.marker.castShadow = true
    this.ring.add(this.marker)

    for (const v of [100, 200, 300, 400, 500]) {
      const el = document.createElement('div')
      el.className = 'ring-tick'
      el.textContent = String(v)
      const tick = new CSS2DObject(el)
      tick.position.copy(this.ringPoint(v, RING_RADIUS + 0.95))
      this.ring.add(tick)
    }
    this.scene.add(this.ring)
  }

  /** Fleet histogram as bars around the ring. */
  setFleet(fleet: number[]) {
    const bins = this.ringBars.count
    const counts = new Array(bins).fill(0)
    for (const v of fleet) {
      const k = Math.min(bins - 1, Math.max(0, Math.floor(((v - CO2_MIN) / (CO2_MAX - CO2_MIN)) * bins)))
      counts[k]++
    }
    const max = Math.max(...counts)
    const m = new THREE.Matrix4()
    counts.forEach((count, i) => {
      const co2 = CO2_MIN + ((i + 0.5) / bins) * (CO2_MAX - CO2_MIN)
      const p = this.ringPoint(co2, RING_RADIUS)
      const h = 0.04 + (count / max) * 0.8
      m.compose(
        new THREE.Vector3(p.x, h / 2, p.z),
        new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), co2Angle(co2)),
        new THREE.Vector3(1, h, 1),
      )
      this.ringBars.setMatrixAt(i, m)
    })
    this.ringBars.instanceMatrix.needsUpdate = true
  }

  private buildSmoke() {
    const count = 160
    this.smoke = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ roughness: 1, flatShading: true }), count)
    this.smoke.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    this.smoke.frustumCulled = false
    for (let i = 0; i < count; i++) {
      this.particles.push({ alive: false, pos: new THREE.Vector3(), vel: new THREE.Vector3(), age: 0, life: 1, size: 0 })
      this.smoke.setColorAt(i, new THREE.Color('#ffffff'))
    }
    this.scene.add(this.smoke)
  }

  private buildLineup() {
    BODIES.forEach((body, i) => {
      const rig = this.lineup[i]
      const x = (i - 2) * 4.2
      rig.root.position.set(x, 0, 0.6)
      rig.root.rotation.y = -0.35
      rig.root.userData.body = body
      this.lineupGroup.add(rig.root)
      const column = new THREE.Mesh(new THREE.BoxGeometry(0.7, 1, 0.7), std('#e9c46a'))
      column.position.set(x + 1.1, 0, -2.6)
      column.castShadow = true
      this.columns.push(column)
      this.lineupGroup.add(column)
      const el = document.createElement('div')
      el.className = 'column-label'
      const label = new CSS2DObject(el)
      label.visible = false
      this.columnLabels.push(label)
      this.lineupGroup.add(label)
    })
  }

  // ---- motion -----------------------------------------------------------

  private tween(duration: number, update: (k: number) => void) {
    if (REDUCED) return update(1)
    this.tweens.push({ start: performance.now(), duration, update })
  }

  private squash(object: THREE.Object3D) {
    this.tween(520, (k) => {
      const y = k < 0.35 ? 1 - 0.3 * Math.sin((k / 0.35) * Math.PI * 0.5) : 0.7 + 0.3 * ease((k - 0.35) / 0.65) + 0.08 * Math.sin(((k - 0.35) / 0.65) * Math.PI)
      object.scale.set(1, y, 1)
    })
  }

  private flash(object: THREE.Object3D) {
    const mats: THREE.MeshStandardMaterial[] = []
    object.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined
      if (m && 'emissive' in m && !mats.includes(m)) mats.push(m)
    })
    const base = mats.map((m) => ({ color: m.emissive.clone(), intensity: m.emissiveIntensity }))
    const glow = new THREE.Color('#f2c14e')
    this.tween(1200, (k) => {
      const pulse = Math.sin(k * Math.PI * 3) * (1 - k)
      mats.forEach((m, i) => {
        m.emissive.copy(base[i].color).lerp(glow, Math.max(0, pulse))
        m.emissiveIntensity = base[i].intensity + Math.max(0, pulse) * 0.8
      })
    })
  }

  private deltaLabel(object: THREE.Object3D, delta: number) {
    if (Math.abs(delta) < 0.05) return
    // CSS2DRenderer positions the outer element with a transform; the animation runs on the inner one.
    const el = document.createElement('div')
    const inner = document.createElement('span')
    inner.className = `delta-label ${delta > 0 ? 'is-up' : 'is-down'}`
    inner.textContent = `${signed(delta)} g/km`
    el.appendChild(inner)
    const label = new CSS2DObject(el)
    const world = new THREE.Box3().setFromObject(object).getCenter(new THREE.Vector3())
    label.position.copy(world).add(new THREE.Vector3(0, 0.5, 0))
    this.scene.add(label)
    window.setTimeout(() => {
      this.scene.remove(label)
      el.remove()
    }, 1900)
  }

  private flyTo(part: Part) {
    const views: Record<Part, [number, number, number]> = {
      body: [7.5, 4.6, 8.5],
      engine: [4.2, 4.8, 7.8],
      gearbox: [8.0, 4.8, 3.4],
      hybrid: [6.2, 4.8, -5.8],
      fuel: [8.4, 4.0, 3.4],
      exhaust: [3.5, 3.2, -8.5],
    }
    const target = part === 'exhaust' ? new THREE.Vector3(0, 0.6, -1.5) : new THREE.Vector3(0, 0.7, 0)
    this.moveCamera(new THREE.Vector3(...views[part]), target)
  }

  private moveCamera(position: THREE.Vector3, target: THREE.Vector3) {
    const p0 = this.camera.position.clone()
    const t0 = this.controls.target.clone()
    this.tween(900, (k) => {
      const e = ease(k)
      this.camera.position.lerpVectors(p0, position, e)
      this.controls.target.lerpVectors(t0, target, e)
    })
  }

  private onPointer = (event: PointerEvent) => {
    if (this.mode !== 'lineup' || !this.options.onSelectBody) return
    const rect = this.renderer.domElement.getBoundingClientRect()
    const ndc = new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1)
    this.raycaster.setFromCamera(ndc, this.camera)
    const hit = this.raycaster.intersectObjects(this.lineup.map((r) => r.root), true)[0]
    if (!hit) return
    let o: THREE.Object3D | null = hit.object
    while (o && !o.userData.body) o = o.parent
    if (o) this.options.onSelectBody(o.userData.body as Body)
  }

  private emitSmoke(dt: number) {
    if (this.mode !== 'single' || !this.rig.config) return
    const rate = THREE.MathUtils.clamp((this.co2 - 80) / 12, 2, 40) // particles per second
    this.emitCarry += rate * dt
    const outlets = this.rig.exhaust.children
    while (this.emitCarry >= 1 && outlets.length) {
      this.emitCarry -= 1
      const p = this.particles.find((q) => !q.alive)
      if (!p) break
      const outlet = outlets[Math.floor(Math.random() * outlets.length)]
      outlet.getWorldPosition(p.pos)
      p.pos.z -= 0.15
      p.vel.set((Math.random() - 0.5) * 0.3, 0.35 + Math.random() * 0.3, -0.9 - Math.random() * 0.5)
      p.age = 0
      p.life = 1.6 + Math.random() * 0.8
      p.size = 0.1 + (this.co2 / 500) * 0.18
      p.alive = true
    }
    const m = new THREE.Matrix4()
    const color = smokeColor(this.co2)
    this.particles.forEach((p, i) => {
      if (p.alive) {
        p.age += dt
        if (p.age > p.life) p.alive = false
        p.pos.addScaledVector(p.vel, dt)
        p.vel.multiplyScalar(1 - dt * 0.6)
      }
      const k = p.age / p.life
      const s = p.alive ? p.size * (1 + k * 2.2) * (1 - k * k) : 0
      m.compose(p.pos, new THREE.Quaternion(), new THREE.Vector3(s, s, s))
      this.smoke.setMatrixAt(i, m)
      this.smoke.setColorAt(i, color)
    })
    this.smoke.instanceMatrix.needsUpdate = true
    if (this.smoke.instanceColor) this.smoke.instanceColor.needsUpdate = true
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.container
    if (!w || !h) return
    this.renderer.setSize(w, h)
    this.labels.setSize(w, h)
    this.composer.setSize(w, h)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  private loop = () => {
    this.frame = requestAnimationFrame(this.loop)
    const now = performance.now()
    const dt = Math.min((now - this.lastFrame) / 1000, 0.05)
    this.lastFrame = now
    this.tweens = this.tweens.filter((t) => {
      const k = Math.min(1, (now - t.start) / t.duration)
      t.update(k)
      return k < 1
    })

    const targetOpacity = this.xrayLocked || now < this.xrayUntil ? 0.22 : 1
    if (Math.abs(this.opacity - targetOpacity) > 0.01) {
      this.opacity += (targetOpacity - this.opacity) * Math.min(1, dt * 8)
      this.rig.setOpacity(this.opacity)
    }
    if (this.rig.config?.HYBRID) {
      const m = this.rig.battery.material as THREE.MeshStandardMaterial
      m.emissiveIntensity = 0.7 + Math.sin(now / 300) * 0.3
      this.rig.motor.rotation.x += dt * 3
    }
    this.rig.gearbox.children.forEach((g, i) => (g.rotation.x += dt * (1.5 + i * 0.4)))

    this.emitSmoke(dt)
    this.controls.update()
    if (this.pixelSize > 1) this.composer.render()
    else this.renderer.render(this.scene, this.camera)
    this.labels.render(this.scene, this.camera)
  }
}
