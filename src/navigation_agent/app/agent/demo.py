"""固定观察和决策的离线接入示例；不评估真实视觉或导航能力。"""

import base64
import copy

from navigation_agent.contracts.agent_backend import make_observation

# 固定 JPEG 仅用于展示接口；模型响应不会识别这张图片。
JPEG = base64.b64decode(
    '/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIx'
    'wcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy'
    'MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAIAAgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAw'
    'QFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkK'
    'FhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmq'
    'KjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEB'
    'AQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRob'
    'HBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOE'
    'hYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+P'
    'n6/9oADAMBAAIRAxEAPwD26iiisSz/2Q=='
)


class DemoBackend:
    look_angle_deg = 70.0
    telemetry = None

    def __init__(self):
        self.actions = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def configure_history(self, enabled):
        # 示例每段立即返回 model_stop，没有可延续的模型历史。
        self.preserve_history = enabled

    def require_control(self):
        pass

    async def observe(self):
        return JPEG, make_observation(
            pose={'x': 0.0, 'y': 0.0, 'yaw': 0.0}, blocked=False, camera_stamp_ns=1)

    async def look(self, direction):
        self.actions.append(('look', direction))
        return {'reason': 'turned', 'direction': direction}

    async def orient(self, heading_deg):
        self.actions.append(('orient', heading_deg))
        return {'reason': 'oriented', 'heading_deg': heading_deg}

    async def navigate(self, instruction, *, seconds):
        self.actions.append(('navigate', instruction, seconds))
        return {'reason': 'model_stop', 'arrival_verified': False}

    async def recover(self):
        return {'reason': 'recovery_unavailable'}

    async def hold(self):
        pass


class DemoModel:
    def __init__(self):
        self.requests = []
        self.responses = [
            {'action': 'look', 'direction': 'left', 'reason': '离线示例：执行一次观察动作'},
            {'action': 'finish', 'reason': '离线示例结束',
             'evidence': '固定测试响应，仅证明软件闭环，不代表真实目标识别'},
        ]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def complete(self, body):
        import json
        self.requests.append(copy.deepcopy(body))
        content = body['messages'][1]['content']
        context = json.loads(content if isinstance(content, str) else content[0]['text'])
        if 'original_goal' in context:
            output = {'instruction': 'Run the offline integration example.'}
        elif context.get('operation') == 'verify_completion':
            output = {name: {'status': 'yes', 'evidence': '固定离线核验响应，不代表真实视觉识别'}
                      for name in ('identity', 'arrival', 'facing')}
        else:
            output = self.responses.pop(0)
        return {'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps(output, ensure_ascii=False)}}],
            'usage': {'total_tokens': 0}}
