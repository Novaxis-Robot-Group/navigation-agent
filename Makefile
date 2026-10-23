SHELL := /bin/bash
.DEFAULT_GOAL := help
.PHONY: help configure mujoco go2 go2-client mujoco-setup go2-setup go2-build models-setup models lightnav yolo mujoco-agent mujoco-sim go2-agent mujoco-check go2-check

help:
	@printf '%s\n' \
	  'MuJoCo 首次准备：make mujoco-setup；在 deploy/.env 填写模型密钥' \
	  'MuJoCo 日常启动：make mujoco（自动检查连接并显示页面地址）' \
	  'Go2 首次准备：make configure；填写本机选项和模型密钥' \
	  'Go2 主机：make go2-setup → make models-setup → make go2' \
	  'Go2 本体：make go2-build → make go2-client' \
	  '共享服务：make models / lightnav / yolo（已有容器直接复用）' \
	  '单独启动：make mujoco-agent / mujoco-sim / go2-agent' \
	  '连接检查：make mujoco-check / go2-check' \
	  '前台会话按 Ctrl+C 退出；共享模型容器保持运行。详见 README。'

configure:
	@bash deploy/manage.sh configure
mujoco go2 go2-client models-setup models lightnav yolo mujoco-agent mujoco-sim go2-agent mujoco-check go2-check:
	@bash deploy/manage.sh $@
mujoco-setup:
	@bash deploy/manage.sh mujoco-setup
go2-setup:
	@bash deploy/manage.sh agent-setup
go2-build:
	@bash deploy/manage.sh go2-build
