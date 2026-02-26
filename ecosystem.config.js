module.exports = {
  apps: [
    {
      name: "ai-trader",
      script: "main.py",
      interpreter: "/home/ubuntu/ai-trader/.venv/bin/python",
      cwd: "/home/ubuntu/ai-trader",
      env: {
        PYTHONPATH: "/home/ubuntu/ai-trader",
      },
      // 자동 재시작
      autorestart: true,
      max_restarts: 10,
      restart_delay: 5000,

      // 로그
      log_date_format: "YYYY-MM-DD HH:mm:ss",
      error_file: "/home/ubuntu/ai-trader/logs/pm2-error.log",
      out_file: "/home/ubuntu/ai-trader/logs/pm2-out.log",
      merge_logs: true,

      // 리소스 모니터링
      max_memory_restart: "500M",

      // 크론 — 장 시간 외 자동 중지/시작은 scheduler가 내부적으로 처리
    },
  ],
};
