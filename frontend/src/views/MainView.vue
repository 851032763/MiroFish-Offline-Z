<template>
  <div class="main-view">
    <!-- 头部 -->
    <header class="app-header">
      <div class="header-left">
        <div class="brand" @click="router.push('/')">MIROFISH OFFLINE</div>
      </div>
      
      <div class="header-center">
        <div class="view-switcher">
          <button 
            v-for="mode in ['graph', 'split', 'workbench']" 
            :key="mode"
            class="switch-btn"
            :class="{ active: viewMode === mode }"
            @click="viewMode = mode"
          >
            {{ { graph: '图谱', split: '分屏', workbench: '工作台' }[mode] }}
          </button>
        </div>
      </div>

      <div class="header-right">
        <div class="workflow-step">
          <span class="step-num">步骤 {{ currentStep }}/5</span>
          <span class="step-name">{{ stepNames[currentStep - 1] }}</span>
        </div>
        <div class="step-divider"></div>
        <span class="status-indicator" :class="statusClass">
          <span class="dot"></span>
          {{ statusText }}
        </span>
      </div>
    </header>

    <!-- 主内容区域 -->
    <main class="content-area">
      <!-- 左侧面板：图谱 -->
      <div class="panel-wrapper left" :style="leftPanelStyle">
        <GraphPanel 
          :graphData="graphData"
          :loading="graphLoading"
          :currentPhase="currentPhase"
          @refresh="refreshGraph"
          @toggle-maximize="toggleMaximize('graph')"
        />
      </div>

      <!-- 右侧面板：步骤组件 -->
      <div class="panel-wrapper right" :style="rightPanelStyle">
        <!-- 步骤 1：图谱构建 -->
        <Step1GraphBuild 
          v-if="currentStep === 1"
          :currentPhase="currentPhase"
          :projectData="projectData"
          :ontologyProgress="ontologyProgress"
          :buildProgress="buildProgress"
          :graphData="graphData"
          :systemLogs="systemLogs"
          @next-step="handleNextStep"
        />
        <!-- 步骤 2：环境配置 -->
        <Step2EnvSetup
          v-else-if="currentStep === 2"
          :projectData="projectData"
          :graphData="graphData"
          :systemLogs="systemLogs"
          @go-back="handleGoBack"
          @next-step="handleNextStep"
          @add-log="addLog"
        />
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import GraphPanel from '../components/GraphPanel.vue'
import Step1GraphBuild from '../components/Step1GraphBuild.vue'
import Step2EnvSetup from '../components/Step2EnvSetup.vue'
import { generateOntology, getProject, buildGraph, getTaskStatus, getGraphData } from '../api/graph'
import { getPendingUpload, clearPendingUpload } from '../store/pendingUpload'

const route = useRoute()
const router = useRouter()

// 布局状态
const viewMode = ref('split') // 图谱 | 分屏 | 工作台

// 步骤状态
const currentStep = ref(1) // 1: 图谱构建, 2: 环境配置, 3: 模拟运行, 4: 报告生成, 5: 交互探索
const stepNames = ['图谱构建', '环境配置', '模拟运行', '报告生成', '交互探索']

// 数据状态
const currentProjectId = ref(route.params.projectId)
const loading = ref(false)
const graphLoading = ref(false)
const error = ref('')
const projectData = ref(null)
const graphData = ref(null)
const currentPhase = ref(-1) // -1: 上传, 0: 本体, 1: 构建, 2: 完成
const ontologyProgress = ref(null)
const buildProgress = ref(null)
const systemLogs = ref([])

// 轮询定时器
let pollTimer = null
let graphPollTimer = null

// --- 计算布局样式 ---
const leftPanelStyle = computed(() => {
  if (viewMode.value === 'graph') return { width: '100%', opacity: 1, transform: 'translateX(0)' }
  if (viewMode.value === 'workbench') return { width: '0%', opacity: 0, transform: 'translateX(-20px)' }
  return { width: '50%', opacity: 1, transform: 'translateX(0)' }
})

const rightPanelStyle = computed(() => {
  if (viewMode.value === 'workbench') return { width: '100%', opacity: 1, transform: 'translateX(0)' }
  if (viewMode.value === 'graph') return { width: '0%', opacity: 0, transform: 'translateX(20px)' }
  return { width: '50%', opacity: 1, transform: 'translateX(0)' }
})

// --- 状态计算属性 ---
const statusClass = computed(() => {
  if (error.value) return 'error'
  if (currentPhase.value >= 2) return 'completed'
  return 'processing'
})

const statusText = computed(() => {
  if (error.value) return '错误'
  if (currentPhase.value >= 2) return '就绪'
  if (currentPhase.value === 1) return '构建图谱中'
  if (currentPhase.value === 0) return '生成本体中'
  return '初始化中'
})

// --- 辅助方法 ---
const addLog = (msg) => {
  const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + '.' + new Date().getMilliseconds().toString().padStart(3, '0')
  systemLogs.value.push({ time, msg })
  // 保留最后100条日志
  if (systemLogs.value.length > 100) {
    systemLogs.value.shift()
  }
}

// --- 布局方法 ---
const toggleMaximize = (target) => {
  if (viewMode.value === target) {
    viewMode.value = 'split'
  } else {
    viewMode.value = target
  }
}

const handleNextStep = (params = {}) => {
  if (currentStep.value < 5) {
    currentStep.value++
    addLog(`进入步骤 ${currentStep.value}: ${stepNames[currentStep.value - 1]}`)

    // 如果从步骤2进入步骤3，记录模拟轮次配置
    if (currentStep.value === 3 && params.maxRounds) {
      addLog(`自定义模拟轮数: ${params.maxRounds}`)
    }
  }
}

const handleGoBack = () => {
  if (currentStep.value > 1) {
    currentStep.value--
    addLog(`返回步骤 ${currentStep.value}: ${stepNames[currentStep.value - 1]}`)
  }
}

// --- 数据逻辑 ---

const initProject = async () => {
  addLog('项目视图已初始化。')
  if (currentProjectId.value === 'new') {
    await handleNewProject()
  } else {
    await loadProject()
  }
}

const handleNewProject = async () => {
  const pending = getPendingUpload()
  if (!pending.isPending || pending.files.length === 0) {
    error.value = '未找到待处理文件。'
    addLog('错误：未找到新项目的待处理文件。')
    return
  }
  
  try {
    loading.value = true
    currentPhase.value = 0
    ontologyProgress.value = { message: '正在上传并分析文档...' }
    addLog('开始生成本体：正在上传文件...')
    
    const formData = new FormData()
    pending.files.forEach(f => formData.append('files', f))
    formData.append('simulation_requirement', pending.simulationRequirement)
    
    const res = await generateOntology(formData)
    if (res.success) {
      clearPendingUpload()
      currentProjectId.value = res.data.project_id
      projectData.value = res.data
      
      router.replace({ name: 'Process', params: { projectId: res.data.project_id } })
      ontologyProgress.value = null
      addLog(`项目 ${res.data.project_id} 本体生成成功`)
      await startBuildGraph()
    } else {
      error.value = res.error || '本体生成失败'
      addLog(`生成本体出错: ${error.value}`)
    }
  } catch (err) {
    error.value = err.message
    addLog(`handleNewProject 异常: ${err.message}`)
  } finally {
    loading.value = false
  }
}

const loadProject = async () => {
  try {
    loading.value = true
    addLog(`正在加载项目 ${currentProjectId.value}...`)
    const res = await getProject(currentProjectId.value)
    if (res.success) {
      projectData.value = res.data
      updatePhaseByStatus(res.data.status)
      addLog(`项目已加载。状态: ${res.data.status}`)
      
      if (res.data.status === 'ontology_generated' && !res.data.graph_id) {
        await startBuildGraph()
      } else if (res.data.status === 'graph_building' && res.data.graph_build_task_id) {
        currentPhase.value = 1
        startPollingTask(res.data.graph_build_task_id)
        startGraphPolling()
      } else if (res.data.status === 'graph_completed' && res.data.graph_id) {
        currentPhase.value = 2
        await loadGraph(res.data.graph_id)
      }
    } else {
      error.value = res.error
      addLog(`加载项目出错: ${res.error}`)
    }
  } catch (err) {
    error.value = err.message
    addLog(`loadProject 异常: ${err.message}`)
  } finally {
    loading.value = false
  }
}

const updatePhaseByStatus = (status) => {
  switch (status) {
    case 'created':
    case 'ontology_generated': currentPhase.value = 0; break;
    case 'graph_building': currentPhase.value = 1; break;
    case 'graph_completed': currentPhase.value = 2; break;
    case 'failed': error.value = '项目失败'; break;
  }
}

const startBuildGraph = async () => {
  try {
    currentPhase.value = 1
    buildProgress.value = { progress: 0, message: '正在启动构建...' }
    addLog('正在启动图构建...')
    
    const res = await buildGraph({ project_id: currentProjectId.value })
    if (res.success) {
      addLog(`图构建任务已启动。任务 ID: ${res.data.task_id}`)
      startGraphPolling()
      startPollingTask(res.data.task_id)
    } else {
      error.value = res.error
      addLog(`启动构建出错: ${res.error}`)
    }
  } catch (err) {
    error.value = err.message
    addLog(`startBuildGraph 异常: ${err.message}`)
  }
}

const startGraphPolling = () => {
  addLog('已开始轮询图数据...')
  fetchGraphData()
  graphPollTimer = setInterval(fetchGraphData, 10000)
}

const fetchGraphData = async () => {
  try {
    // 刷新项目信息以检查 graph_id
    const projRes = await getProject(currentProjectId.value)
    if (projRes.success && projRes.data.graph_id) {
      const gRes = await getGraphData(projRes.data.graph_id)
      if (gRes.success) {
        graphData.value = gRes.data
        const nodeCount = gRes.data.node_count || gRes.data.nodes?.length || 0
        const edgeCount = gRes.data.edge_count || gRes.data.edges?.length || 0
        addLog(`图数据已刷新。节点数: ${nodeCount}, 边数: ${edgeCount}`)
      }
    }
  } catch (err) {
    console.warn('图数据获取错误:', err)
  }
}

const startPollingTask = (taskId) => {
  pollTaskStatus(taskId)
  pollTimer = setInterval(() => pollTaskStatus(taskId), 2000)
}

const pollTaskStatus = async (taskId) => {
  try {
    const res = await getTaskStatus(taskId)
    if (res.success) {
      const task = res.data
      
      // 如果进度消息改变则记录日志
      if (task.message && task.message !== buildProgress.value?.message) {
        addLog(task.message)
      }
      
      buildProgress.value = { progress: task.progress || 0, message: task.message }
      
      if (task.status === 'completed') {
        addLog('图构建任务已完成。')
        stopPolling()
        stopGraphPolling() // 停止轮询，执行最终加载
        currentPhase.value = 2
        
        // 最终加载
        const projRes = await getProject(currentProjectId.value)
        if (projRes.success && projRes.data.graph_id) {
            projectData.value = projRes.data
            await loadGraph(projRes.data.graph_id)
        }
      } else if (task.status === 'failed') {
        stopPolling()
        error.value = task.error
        addLog(`图构建任务失败: ${task.error}`)
      }
    }
  } catch (e) {
    console.error(e)
  }
}

const loadGraph = async (graphId) => {
  graphLoading.value = true
  addLog(`正在加载完整图数据: ${graphId}`)
  try {
    const res = await getGraphData(graphId)
    if (res.success) {
      graphData.value = res.data
      addLog('图数据加载成功。')
    } else {
      addLog(`加载图数据失败: ${res.error}`)
    }
  } catch (e) {
    addLog(`加载图异常: ${e.message}`)
  } finally {
    graphLoading.value = false
  }
}

const refreshGraph = () => {
  if (projectData.value?.graph_id) {
    addLog('已触发手动图刷新。')
    loadGraph(projectData.value.graph_id)
  }
}

const stopPolling = () => {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

const stopGraphPolling = () => {
  if (graphPollTimer) {
    clearInterval(graphPollTimer)
    graphPollTimer = null
    addLog('图轮询已停止。')
  }
}

onMounted(() => {
  initProject()
})

onUnmounted(() => {
  stopPolling()
  stopGraphPolling()
})
</script>

<style scoped>
.main-view {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #FFF;
  overflow: hidden;
  font-family: 'Space Grotesk', 'Noto Sans SC', system-ui, sans-serif;
}

/* 头部 */
.app-header {
  height: 60px;
  border-bottom: 1px solid #EAEAEA;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: #FFF;
  z-index: 100;
  position: relative;
}

.header-center {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
}

.brand {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 800;
  font-size: 18px;
  letter-spacing: 1px;
  cursor: pointer;
}

.view-switcher {
  display: flex;
  background: #F5F5F5;
  padding: 4px;
  border-radius: 6px;
  gap: 4px;
}

.switch-btn {
  border: none;
  background: transparent;
  padding: 6px 16px;
  font-size: 12px;
  font-weight: 600;
  color: #666;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.switch-btn.active {
  background: #FFF;
  color: #000;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #666;
  font-weight: 500;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.workflow-step {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.step-num {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  color: #999;
}

.step-name {
  font-weight: 700;
  color: #000;
}

.step-divider {
  width: 1px;
  height: 14px;
  background-color: #E0E0E0;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #CCC;
}

.status-indicator.processing .dot { background: #FF5722; animation: pulse 1s infinite; }
.status-indicator.completed .dot { background: #4CAF50; }
.status-indicator.error .dot { background: #F44336; }

@keyframes pulse { 50% { opacity: 0.5; } }

/* 内容区域 */
.content-area {
  flex: 1;
  display: flex;
  position: relative;
  overflow: hidden;
}

.panel-wrapper {
  height: 100%;
  overflow: hidden;
  transition: width 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), opacity 0.3s ease, transform 0.3s ease;
  will-change: width, opacity, transform;
}

.panel-wrapper.left {
  border-right: 1px solid #EAEAEA;
}
</style>
