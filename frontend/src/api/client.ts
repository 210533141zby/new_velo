import axios, { AxiosError } from 'axios';
import { ElMessage } from 'element-plus';

type ApiErrorPayload = {
  detail?: string | { message?: string } | Array<{ msg?: string }>;
  message?: string;
};

function resolveErrorMessage(error: AxiosError<ApiErrorPayload>) {
  const payload = error.response?.data;

  if (error.code === 'ECONNABORTED' || error.message.toLowerCase().includes('timeout')) {
    return '请求超时，请稍后重试';
  }

  if (typeof payload?.detail === 'string') {
    return payload.detail;
  }

  if (Array.isArray(payload?.detail) && payload.detail[0]?.msg) {
    return payload.detail[0].msg;
  }

  if (payload?.message) {
    return payload.message;
  }

  return error.message || '请求失败';
}

const client = axios.create({
  baseURL: '/api/v1',
  timeout: 15000,
});

client.interceptors.response.use(
  (response) => response.data,
  (error: AxiosError<ApiErrorPayload>) => {
    if (error.code !== 'ERR_CANCELED') {
      console.error('API Error:', error);
      ElMessage.error(resolveErrorMessage(error));
    }

    return Promise.reject(error);
  },
);

export default client;
