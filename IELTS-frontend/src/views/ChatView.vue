<template>
  <div class="chat-container">
    <div class="messages">
      <div v-for="(msg, idx) in messages" :key="idx" :class="msg.role">
        <strong v-if="msg.role === 'user'">You:</strong>
        <strong v-else>AI:</strong>
        <p>{{ msg.text }}</p>
      </div>
    </div>

    <form @submit.prevent="sendMessage" class="input-area">
      <input v-model="userInput" placeholder="Ask me something..." />
      <button @click="toggleRecording">
        <span v-if="!isRecording">🎤</span>
        <span v-else>🛑</span>
      </button>
      <button :disabled="loading" type="submit">{{ loading ? 'Sending...' : 'Send' }}</button>
    </form>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const userInput = ref('')
const messages = ref([])
const loading = ref(false)
const isRecording = ref(false)

let recognition = null

onMounted(() => {
  if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    recognition = new SpeechRecognition()
    recognition.lang = 'en-US'
    recognition.interimResults = false
    recognition.maxAlternatives = 1

    recognition.onstart = () => {
      isRecording.value = true
      console.log("🎤 开始录音")
    }

    recognition.onresult = (event) => {
      console.log("📥 onresult 触发了！");
      const transcript = event.results[0][0].transcript
      console.log("✅ 识别结果：", transcript)
      userInput.value = transcript
    }

    recognition.onerror = (event) => {
      console.error("❌ 语音识别错误：", event.error)
    }

    recognition.onend = () => {
      isRecording.value = false
      console.log("🛑 录音结束")
    }

  } else {
    alert("Your browser does not support Web Speech API.")
  }
})

const toggleRecording = () => {
  if (!recognition) return
  if (isRecording.value) {
    recognition.stop()
  } else {
    recognition.start()
  }
}

// ✅ 调用后端发送消息
const sendMessage = async () => {
  if (!userInput.value.trim()) return
  messages.value.push({ role: 'user', text: userInput.value })
  loading.value = true
  try {
    const res = await axios.post('http://localhost:8080/api/chat', { text: userInput.value })
    messages.value.push({ role: 'bot', text: res.data.response })
  } catch (err) {
    messages.value.push({ role: 'bot', text: "⚠️ Failed to connect to backend." })
  }
  loading.value = false
  userInput.value = ''
}
</script>

<style scoped>
.chat-container {
  max-width: 700px;
  margin: 2rem auto;
  display: flex;
  flex-direction: column;
}
.messages {
  border: 1px solid #ccc;
  padding: 1rem;
  height: 400px;
  overflow-y: auto;
  background: #f9f9f9;
}
.user p {
  text-align: right;
  color: #333;
}
.bot p {
  text-align: left;
  color: #0066cc;
}
.input-area {
  display: flex;
  margin-top: 1rem;
}
input {
  flex: 1;
  padding: 0.5rem;
  font-size: 1rem;
}
button {
  padding: 0.5rem 1rem;
  margin-left: 0.5rem;
}
</style>
