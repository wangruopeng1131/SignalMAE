#!/bin/sh

# 检查是否传递了参数，如果没有则打印错误信息并退出脚本
if [ $# -eq 0 ]; then
    echo "错误：没有传递参数！"
    exit 1
fi

# 检查要执行的命令是否为启动或停止命令
if [ "$1" = "start" ]; then
  shift
  # 启动命令
  # 拼接所有设备ID，用逗号分隔
  DEVICES=$(echo "$@" | tr ' ' ',')

  # 检查是否传递了设备ID参数，如果没有，则使用默认值
  if [ -z "$DEVICES" ]; then
    DEVICES="$DEFAULT_DEVICES"
  fi

  # 设置环境变量CUDA_VISIBLE_DEVICES的值
  export CUDA_VISIBLE_DEVICES="$DEVICES"
  export OMP_NUM_THREADS=30
  # 获取传入参数的数量
  NUM_ARGS=$#

  # 打印传入参数的数量
  echo "NUM GPUs：$NUM_ARGS"
  echo "GPU ID：$DEVICES"
  torchrun --standalone --nproc_per_node $NUM_ARGS pretrain_mae.py
  echo "start......"
elif [ "$1" = "stop" ]; then

  PID=$(ps -ef | grep wangruopeng | grep torchrun | grep pretrain_mae.py | grep -v grep | awk '{print $2}')

  if [ -n "$PID" ]; then
    echo "torchrun is exist"
  else
      echo "torchrun is not exist"
      exit 1
  fi

  # 停止命令
  kill -9 $PID
  echo "stop......"
else
  echo "error: please input start or stop!"
  exit 1
fi



