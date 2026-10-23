"""真实任务的固定观察与动作回归，包括 9 月 24 日导航和 10 月 12 日原地往返。"""

import copy
import json
import unittest

from navigation_agent.agent.schemas import Decision, ActionReview
from tests.config_support import execution_state


# 只保留回归所需的位姿、观察编号和原模型提案，不包含图片或运行目录。
CASES = json.loads(r'''[
  {
    "step": 8,
    "start": {
      "id": "P1",
      "observer_position": [
        -0.4201247990131378,
        0.29580870270729065
      ],
      "heading_deg": -3.165599858374342,
      "image": "images/006.jpg",
      "step": 6
    },
    "arrival": {
      "id": "P2",
      "observer_position": [
        -1.697519063949585,
        1.6294437646865845
      ],
      "heading_deg": 137.34823322500236,
      "image": "images/007.jpg",
      "step": 7
    },
    "current": {
      "id": "P2",
      "observer_position": [
        -1.6512879133224487,
        1.6125203371047974
      ],
      "heading_deg": -153.27693994014157,
      "image": "images/008.jpg",
      "step": 8
    },
    "previous_navigation": {
      "action": "navigate",
      "reason": "环视完成，未见沙发或人。当前朝向约-3度，左前方可见玻璃幕墙附近的开放地面和通道。选择该方向探索，可能发现新的区域或通向其他房间的入口。路径为灰色地毯，无人员工位障碍。",
      "instruction": "Walk straight toward the open floor area near the glass wall on your left and stop.",
      "evidence": "",
      "navigation_check": {
        "target_area": "allowed",
        "path_area": "allowed",
        "passage": "clear",
        "evidence": "终点是左前方玻璃墙附近的开放地面，路径为灰色地毯，无在座人员工位障碍，无需穿越玻璃或关闭门"
      },
      "scene": {
        "landmarks": [
          "白色双门储物柜",
          "落地窗及窗外城市建筑",
          "黑色三脚架",
          "办公打印机/复印机",
          "粉色地垫",
          "黄黑警示胶带",
          "灰色办公椅",
          "电脑显示器"
        ],
        "visible_ground": "前景有粉色地垫和警示胶带，左侧地面通向窗户方向，右侧被设备和三脚架部分遮挡",
        "occluded": [
          "柜子后方空间",
          "三脚架和打印机后方区域"
        ],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      },
      "candidates": [
        {
          "id": "new_1",
          "kind": "open_passage",
          "description": "左前方玻璃幕墙附近的开放地面",
          "evidence": "当前图左侧可见玻璃墙和灰色地毯延伸的通道，路径清晰可接近",
          "expected_new_view": "看清玻璃墙附近是否有其他出口或连接区域，寻找可能的沙发位置"
        }
      ],
      "candidate_id": "new_1"
    },
    "result": {
      "reason": "timeout",
      "instruction": "Walk straight toward the open floor area near the glass wall on your left and stop.",
      "motion": {
        "samples": 308,
        "displacement_m": 1.847,
        "path_length_m": 2.448,
        "absolute_turn_deg": 184.1,
        "turn_direction_changes": 19,
        "blocked_samples": 0,
        "interpretation": "Motion statistics only; doorway crossing and goal arrival require visual verification."
      },
      "arrival_verified": false,
      "attempts": [
        {
          "attempt": 1,
          "reason": "timeout"
        }
      ],
      "candidate_id": "C1"
    },
    "proposal": {
      "action": "navigate",
      "candidate_id": "new_1",
      "instruction": "Walk straight toward the open floor area near the glass wall ahead and stop.",
      "reason": "导航到达P2新位置，当前朝向约-153度。前方可见玻璃幕墙前的开放地面区域，该方向尚未探索。接近后可看清玻璃墙附近是否有其他出口或连接区域，寻找可能的沙发位置。路径为灰色地毯和粉色地垫，无遮挡。",
      "scene": {
        "landmarks": [
          "玻璃幕墙",
          "白色柜台/接待台",
          "粉色地垫",
          "灰色地毯",
          "绿色安全出口标志",
          "红色消防设备"
        ],
        "visible_ground": "前景为灰色地毯和粉色地垫边缘，中景有白色柜台，左侧地面通向玻璃墙方向",
        "occluded": [
          "白色柜台后方区域",
          "玻璃幕墙内部"
        ],
        "target_seen": "no",
        "place_label": ""
      },
      "candidates": [
        {
          "id": "new_1",
          "kind": "open_passage",
          "description": "玻璃幕墙前的开放地面区域",
          "evidence": "当前图前方可见玻璃墙和灰色地毯延伸的通道，路径清晰可接近",
          "expected_new_view": "看清玻璃墙附近是否有其他出口或连接区域，寻找可能的沙发位置"
        }
      ],
      "previous_action_review": {
        "outcome": "progress",
        "view_change": "new_area",
        "evidence": "从P1导航到P2，位移1.85m，进入新的观察位置，视野从储物柜方向转向玻璃幕墙和白色柜台区域"
      },
      "navigation_check": {
        "target_area": "allowed",
        "path_area": "allowed",
        "passage": "clear",
        "evidence": "终点为玻璃幕墙前的开放地面，路径为灰色地毯和粉色地垫，无玻璃或关闭门障碍，路径畅通"
      }
    }
  },
  {
    "step": 11,
    "start": {
      "id": "P2",
      "observer_position": [
        -1.6372175216674805,
        1.5254120826721191
      ],
      "heading_deg": 144.17764522284838,
      "image": "images/009.jpg",
      "step": 9
    },
    "arrival": {
      "id": "P3",
      "observer_position": [
        -4.3228607177734375,
        3.5900726318359375
      ],
      "heading_deg": 142.07412410292602,
      "image": "images/010.jpg",
      "step": 10
    },
    "current": {
      "id": "P3",
      "observer_position": [
        -4.271947860717773,
        3.6335067749023438
      ],
      "heading_deg": -148.75585968755624,
      "image": "images/011.jpg",
      "step": 11
    },
    "previous_navigation": {
      "action": "navigate",
      "reason": "环视结束，当前图前方可见玻璃幕墙和白色柜台之间的开放地面通道，路径清晰。接近该区域可看清玻璃墙附近是否有其他出口或连接区域，寻找可能的沙发位置。",
      "instruction": "Walk straight toward the open floor area in front of the white counter ahead and stop.",
      "evidence": "",
      "navigation_check": {
        "target_area": "allowed",
        "path_area": "allowed",
        "passage": "clear",
        "evidence": "本次终点为白色柜台前方的灰色地毯空地，路径从当前位置到柜台前地面连续可见，无玻璃或关闭门阻挡，不涉及在座人员工位禁区"
      },
      "scene": {
        "landmarks": [
          "玻璃幕墙",
          "白色柜台/接待台",
          "绿色安全出口标志",
          "红色消防设备",
          "灰色地毯",
          "粉色地垫边缘"
        ],
        "visible_ground": "前景为灰色地毯，中景有白色柜台，左侧地面通向玻璃墙方向，通道较开阔",
        "occluded": [
          "白色柜台后方区域",
          "玻璃幕墙内部"
        ],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      },
      "candidates": [
        {
          "id": "new_1",
          "kind": "open_passage",
          "description": "玻璃幕墙与白色柜台之间的开放地面通道",
          "evidence": "当前图正前方可见玻璃幕墙和白色柜台，两者之间地面为灰色地毯，路径连续无遮挡，可安全接近",
          "expected_new_view": "接近后看清玻璃墙附近是否有其他出口、门洞或连接区域，确认是否存在沙发所在的休息区"
        }
      ],
      "candidate_id": "new_1"
    },
    "result": {
      "reason": "model_stop",
      "instruction": "Walk straight toward the open floor area in front of the white counter ahead and stop.",
      "motion": {
        "samples": 257,
        "displacement_m": 3.386,
        "path_length_m": 3.589,
        "absolute_turn_deg": 67.6,
        "turn_direction_changes": 29,
        "blocked_samples": 0,
        "interpretation": "Motion statistics only; doorway crossing and goal arrival require visual verification."
      },
      "arrival_verified": false,
      "attempts": [
        {
          "attempt": 1,
          "reason": "model_stop"
        }
      ],
      "candidate_id": "C2"
    },
    "proposal": {
      "action": "navigate",
      "candidate_id": "new_1",
      "instruction": "Walk straight toward the open corridor on your left and stop.",
      "reason": "当前图左侧可见一条延伸的开放走廊，地面为浅色地砖，路径清晰无遮挡。之前探索的区域（P1、P2）主要是办公区和玻璃幕墙附近，未发现沙发。这条走廊可能通向休息区或其他房间，值得探索以寻找坐在沙发上的人。",
      "scene": {
        "landmarks": [
          "白色墙壁",
          "绿色安全出口标志",
          "玻璃幕墙",
          "金属门框",
          "浅色地砖"
        ],
        "visible_ground": "前景为浅色地砖，左侧走廊地面连续可见，右侧被白色墙壁和玻璃隔断部分遮挡",
        "occluded": [
          "玻璃幕墙内部区域",
          "走廊尽头空间"
        ],
        "target_seen": "no",
        "place_label": ""
      },
      "candidates": [
        {
          "id": "new_1",
          "kind": "open_passage",
          "description": "左侧延伸的开放走廊",
          "evidence": "当前图左侧可见一条清晰的走廊，地面为浅色地砖，路径连续无遮挡，可安全接近",
          "expected_new_view": "沿走廊前进，查看是否有休息区、沙发或其他连接区域，寻找坐在沙发上的人"
        }
      ],
      "previous_action_review": {
        "outcome": "progress",
        "view_change": "new_area",
        "evidence": "从P2导航到P3，净位移3.39m，进入玻璃幕墙附近的区域，视角变化明显，看到新的走廊方向"
      },
      "navigation_check": {
        "target_area": "allowed",
        "path_area": "allowed",
        "passage": "clear",
        "evidence": "本次终点是左侧走廊内的开放地面，路径为浅色地砖，连续可见，无需穿越玻璃或关闭门，可安全接近"
      }
    }
  }
]''')


class RecordedExplorationTests(unittest.TestCase):
    def test_original_new_area_navigation_waits_for_scan_without_changing_state(self):
        for case in CASES:
            with self.subTest(step=case['step']):
                state = execution_state(look_angle_deg=75, navigation_policy='go2_office')
                state.observe(case['start'])
                navigation = Decision.model_validate(case['proposal'])
                # 重放实际导航和补看前缀，不执行提案中的任何运动。
                previous = Decision.model_validate(case['previous_navigation'])
                # 日志截取从已完成首轮环视后的导航开始。
                state.scan.complete = True
                previous.navigation_check.space = 'normal'
                selected = state.accept(previous)
                state.navigation_started(selected)
                state.record_result(selected, case['result'])
                state.last_action = {'step': case['start']['step'], 'action': 'navigate',
                                     'candidate_id': selected}
                state.observe(case['arrival'])
                state.accept(Decision(action='look', direction='left', reason='检查旁边的遮挡'))
                state.last_action = {'step': case['arrival']['step'], 'action': 'look'}
                state.observe(case['current'])
                state.scan.look_count = 100
                # 本用例只复核环视状态，空间报告单独回归。
                navigation.navigation_check.space = 'normal'
                options = state.action_options()
                before = state.snapshot()
                state.resolve(navigation)
                self.assertEqual(state.snapshot(), before)
                self.assertEqual(state.action_options(), options)
                self.assertFalse(options['scan_required'])
                self.assertEqual(state.scan.current_view_id, f'O{case["step"]}')

    def test_old_position_orient_still_rejected_and_options_are_explicit(self):
        case = CASES[1]
        state = execution_state(look_angle_deg=75)
        old = dict(CASES[0]['arrival'], image='images/007.jpg', step=7)
        state.observe(old)
        state.observe(case['arrival'])
        state.observe(case['current'])
        self.assertEqual(state.action_options()['orient_view_ids'], ['O10'])
        with self.assertRaisesRegex(ValueError, '当前 action_options'):
            state.resolve(Decision(action='orient', view_id='O7', reason='转回旧位置方向'))
        self.assertEqual(state.action_options()['look_directions'], ['left', 'right'])


# 11:40 找垃圾桶任务的实际观察位姿与提案；只读回放，不执行原动作。
TURNING_TRACE = json.loads(r'''[
  {
    "view": {
      "id": "P1",
      "observer_position": [
        -0.2934701144695282,
        0.004550503566861153
      ],
      "heading_deg": 152.51500357455225,
      "image": "images/001.jpg",
      "step": 1
    },
    "proposal": {
      "action": "look",
      "reason": "当前视角（O1）未见垃圾桶，需要环视寻找目标。先向左观察以扩大视野覆盖范围。",
      "direction": "left"
    }
  },
  {
    "view": {
      "id": "P1",
      "observer_position": [
        -0.2592398524284363,
        0.053693681955337524
      ],
      "heading_deg": -137.3516417286794,
      "image": "images/002.jpg",
      "step": 2
    },
    "proposal": {
      "action": "look",
      "reason": "当前视角（O2）看到办公桌下堆放着纸箱，未见垃圾桶。继续向右观察以扩大视野覆盖范围，寻找目标。",
      "direction": "right"
    }
  },
  {
    "view": {
      "id": "P1",
      "observer_position": [
        -0.2033311426639557,
        0.04190146550536156
      ],
      "heading_deg": 155.0488574676515,
      "image": "images/003.jpg",
      "step": 3
    },
    "proposal": {
      "action": "look",
      "reason": "当前视角（O3）看到办公区域，有办公桌、玻璃幕墙和椅子，但未见垃圾桶。继续向右观察以扩大视野覆盖范围，寻找目标。",
      "direction": "right"
    }
  },
  {
    "view": {
      "id": "P1",
      "observer_position": [
        -0.20560579001903534,
        -0.013747465796768665
      ],
      "heading_deg": 87.82107241621853,
      "image": "images/004.jpg",
      "step": 4
    },
    "proposal": {
      "action": "look",
      "reason": "当前视角(O4)看到办公区域，有办公桌、椅子和水瓶，未见垃圾桶。根据scan信息，还有右侧方向未观察（unobserved_look_directions: [right]），且最大角度间隙达225度，说明视野覆盖不完整。继续向右观察以完成环视，寻找垃圾桶。",
      "direction": "right"
    }
  },
  {
    "view": {
      "id": "P1",
      "observer_position": [
        -0.25491541624069214,
        -0.04740779846906662
      ],
      "heading_deg": 21.419395797068464,
      "image": "images/005.jpg",
      "step": 5
    },
    "proposal": {
      "action": "look",
      "reason": "当前视角（O5）看到办公桌、玻璃幕墙和水瓶，未见垃圾桶。根据scan信息，还有右侧方向未观察（unobserved_look_directions: right），且最大角度缺口为158度。继续向右观察以完成环视覆盖，寻找目标。",
      "direction": "right"
    }
  },
  {
    "view": {
      "id": "P1",
      "observer_position": [
        -0.303293913602829,
        -0.022967325523495674
      ],
      "heading_deg": -44.74458890108984,
      "image": "images/006.jpg",
      "step": 6
    },
    "proposal": {
      "action": "look",
      "reason": "当前视角（O6）未见垃圾桶，但看到左侧白色柜体和右侧三脚架之间的通道。根据scan信息，还有左侧方向未观察（unobserved_look_directions: left）。先向左观察以扩大视野覆盖范围，寻找目标。",
      "direction": "left"
    }
  },
  {
    "view": {
      "id": "P1",
      "observer_position": [
        -0.3568715751171112,
        -0.06070699915289879
      ],
      "heading_deg": 24.037463934699534,
      "image": "images/007.jpg",
      "step": 7
    },
    "proposal": {
      "action": "look",
      "reason": "当前视角（O7）看到办公桌、玻璃幕墙和水瓶，未见垃圾桶。根据scan信息，还有右侧方向未观察（unobserved_look_directions: right），需要继续向右扩大视野覆盖范围以寻找目标。",
      "direction": "right"
    }
  },
  {
    "view": {
      "id": "P1",
      "observer_position": [
        -0.41436025500297546,
        -0.030553191900253296
      ],
      "heading_deg": -44.97121006328963,
      "image": "images/008.jpg",
      "step": 8
    },
    "proposal": {
      "action": "navigate",
      "reason": "当前视角(O8)看到白色柜体和三脚架，以及右侧有办公桌和纸箱的开放区域。虽然之前观察过附近方向，但尚未实际移动探索该通道。接近三脚架附近的空地可以看清该区域的细节，寻找垃圾桶或新的通道入口。",
      "instruction": "Walk straight toward the open floor area near the tripod on your right and stop.",
      "evidence": "",
      "navigation_check": {
        "target_area": "allowed",
        "path_area": "allowed",
        "passage": "clear",
        "evidence": "终点为三脚架旁的灰色地垫空地，路径上无玻璃或关闭门阻挡，可直接通行。"
      },
      "scene": {
        "landmarks": [
          "白色高柜",
          "黑色三脚架",
          "办公桌",
          "棕色纸箱",
          "粉色地垫"
        ],
        "visible_ground": "灰色拼接地垫和粉色地垫",
        "occluded": [
          "白色柜体后方",
          "三脚架后方",
          "办公桌下方"
        ],
        "target_seen": "no",
        "place_label": "P1",
        "objects": [
          {
            "name": "棕色纸箱",
            "description": "位于右侧办公桌下",
            "bearing_deg": -30
          }
        ]
      },
      "candidates": [
        {
          "id": "new_1",
          "kind": "viewpoint",
          "description": "三脚架附近的开放地面",
          "evidence": "当前图可见三脚架右侧有一片灰色地垫覆盖的空地，路径畅通",
          "expected_new_view": "接近后看清该区域是否有垃圾桶，或者是否有通往其他房间的开口"
        }
      ],
      "candidate_id": "new_1"
    }
  }
]''')


class RecordedTurningTests(unittest.TestCase):
    def test_original_back_and_forth_is_rejected_by_fixed_direction(self):
        state = execution_state(look_angle_deg=75)
        first = TURNING_TRACE[0]
        state.observe(first['view'])
        state.accept(Decision.model_validate(first['proposal']))
        state.observe(TURNING_TRACE[1]['view'])
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, '复核'):
            state.resolve(Decision.model_validate(TURNING_TRACE[1]['proposal']))
        self.assertEqual(state.snapshot(), before)
        self.assertEqual(state.action_options()['look_directions'], ['left', 'right'])
        state.resolve(Decision(action='look', direction='left', reason='继续同向环视'))


    def test_new_viewpoint_keeps_memory_and_requires_a_new_scan(self):
        state = execution_state(look_angle_deg=75)
        for row in TURNING_TRACE:
            state.observe(row['view'])
        previous_ids = [view['id'] for view in state.scan.views]
        state.last_action = {'step': 8, 'action': 'navigate'}
        state.observe({
            'id': 'P2', 'observer_position': [.199, -1.271],
            'heading_deg': -40.257, 'step': 9, 'image': 'images/009.jpg',
        })
        self.assertEqual([view['id'] for view in state.scan.views[:-1]], previous_ids)
        self.assertTrue(all(option['repeat_of'] is None
                            for option in state.action_options()['look_options'].values()))
        state.resolve(Decision(action='look', direction='left', reason='补看柜边新出现的开口'))
        state.resolve(Decision.model_validate(TURNING_TRACE[-1]['proposal']))


# 13:40 任务的实际图像位姿与导航提案；保留原提案，不改写执行事实。
LATEST_SCAN_TRACE = json.loads(r'''[
  {
    "view": {
      "id": "P1",
      "step": 1,
      "observer_position": [
        -0.01948612742125988,
        0.03455771878361702
      ],
      "heading_deg": -9.693823192881252,
      "image": "images/001.jpg"
    },
    "decision": {
      "action": "look",
      "reason": "查看右侧尚未观察的通道，寻找银色大垃圾桶",
      "direction": "right",
      "instruction": "",
      "evidence": "",
      "scene": {
        "landmarks": [
          "玻璃幕墙",
          "办公桌",
          "办公椅"
        ],
        "visible_ground": "地毯和拼接地垫",
        "occluded": [],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      },
      "candidates": []
    }
  },
  {
    "view": {
      "id": "P1",
      "step": 2,
      "observer_position": [
        -0.049378424882888794,
        0.07238519191741943
      ],
      "heading_deg": -77.33610088990828,
      "image": "images/002.jpg"
    },
    "decision": {
      "action": "look",
      "reason": "查看右侧尚未观察的通道，寻找银色大垃圾桶。当前视野为办公桌和玻璃幕墙，未见目标，继续向右环视以覆盖盲区。",
      "direction": "right",
      "instruction": "",
      "evidence": "",
      "scene": {
        "landmarks": [
          "玻璃幕墙",
          "办公桌",
          "办公椅"
        ],
        "visible_ground": "地毯",
        "occluded": [],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      },
      "candidates": []
    }
  },
  {
    "view": {
      "id": "P1",
      "step": 3,
      "observer_position": [
        -0.012018488720059395,
        0.11499380320310593
      ],
      "heading_deg": -144.56177454756903,
      "image": "images/003.jpg"
    },
    "decision": {
      "action": "look",
      "reason": "继续向右环视，查看当前右前方的白色高柜右侧及后方区域，寻找银色大垃圾桶。",
      "direction": "right",
      "instruction": "",
      "evidence": "",
      "scene": {
        "landmarks": [
          "办公桌",
          "办公椅",
          "白色高柜",
          "玻璃幕墙"
        ],
        "visible_ground": "拼接地垫和地毯",
        "occluded": [],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      },
      "candidates": []
    }
  },
  {
    "view": {
      "id": "P1",
      "step": 4,
      "observer_position": [
        0.04133014753460884,
        0.09574922174215317
      ],
      "heading_deg": 148.00131997878358,
      "image": "images/004.jpg"
    },
    "decision": {
      "action": "look",
      "reason": "继续向右环视，查看白色高柜右侧及后方区域，寻找银色大垃圾桶。当前视野为三脚架、纸箱和办公区，未见目标。",
      "direction": "right",
      "instruction": "",
      "evidence": "",
      "scene": {
        "landmarks": [
          "三脚架",
          "白色高柜",
          "纸箱",
          "办公桌"
        ],
        "visible_ground": "拼接地垫和地毯",
        "occluded": [],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      },
      "candidates": []
    }
  },
  {
    "view": {
      "id": "P1",
      "step": 5,
      "observer_position": [
        0.05787414312362671,
        0.04401474818587303
      ],
      "heading_deg": 79.91478598682983,
      "image": "images/005.jpg"
    },
    "decision": {
      "action": "navigate",
      "reason": "当前视野中未见目标，但左侧白色高柜前有一块可接近的开放地面。移动到该位置可以改变视角，查看高柜后方或右侧被遮挡的区域，寻找银色大垃圾桶。",
      "instruction": "Walk straight toward the open floor area in front of the white cabinet on your left and stop.",
      "evidence": "",
      "navigation_check": {
        "target_area": "allowed",
        "path_area": "allowed",
        "passage": "clear",
        "evidence": "终点为白色高柜前的空地，属于允许通行的过道区域；从当前位置到该空地的路径没有障碍物，且不需要穿越玻璃或关闭的门。"
      },
      "scene": {
        "landmarks": [
          "办公桌",
          "办公椅",
          "白色高柜",
          "三脚架",
          "纸箱"
        ],
        "visible_ground": "拼接地垫和地毯",
        "occluded": [
          "白色高柜后方",
          "办公桌下方"
        ],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      },
      "candidates": [
        {
          "id": "new_1",
          "kind": "viewpoint",
          "description": "白色高柜前的开放地面",
          "evidence": "当前图左侧可见白色高柜，其前方有一块未被桌椅占据的空地，路径清晰。",
          "expected_new_view": "通过移动位置，看清白色高柜后方及右侧是否藏有银色大垃圾桶。"
        }
      ],
      "candidate_id": "new_1"
    }
  },
  {
    "view": {
      "id": "P1",
      "step": 6,
      "observer_position": [
        0.11052487045526505,
        0.4200216829776764
      ],
      "heading_deg": 156.6526496393163,
      "image": "images/006.jpg"
    },
    "decision": {
      "action": "navigate",
      "reason": "当前视野中，三脚架左侧有一块地毯覆盖的开放地面，通向玻璃幕墙。此前探索了右侧和正前方的柜体区域，尚未检查该侧通道。移动到此处可以看清玻璃门后的情况以及是否有银色垃圾桶藏匿于此。",
      "instruction": "Walk straight toward the open floor area in front of the tripod and stop.",
      "evidence": "",
      "navigation_check": {
        "target_area": "allowed",
        "path_area": "allowed",
        "passage": "clear",
        "evidence": "目的地是三脚架旁的空地，属于办公区过道；路径上只有地毯，没有穿越障碍或在座人员工位。"
      },
      "previous_action_review": {
        "outcome": "progress",
        "view_change": "same_area",
        "evidence": "从 O5 到 O6，机器人移动到了白色高柜前，视角转向了三脚架和玻璃墙方向，看到了新的地标（三脚架）。"
      },
      "scene": {
        "landmarks": [
          "黑色三脚架",
          "白色矮柜",
          "纸箱",
          "玻璃幕墙"
        ],
        "visible_ground": "灰色地毯，局部有粉色地垫",
        "occluded": [
          "玻璃幕墙后的空间",
          "三脚架后方"
        ],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      },
      "candidates": [
        {
          "id": "new_1",
          "kind": "viewpoint",
          "description": "三脚架左侧、玻璃幕墙前的空地",
          "evidence": "当前图左前方可见三脚架立柱旁的地毯区域，路径清晰，无桌椅阻挡。",
          "expected_new_view": "接近后确认玻璃门是否开启，并查看墙角或幕布后是否有目标。"
        }
      ],
      "candidate_id": "new_1"
    }
  }
]''')

class FixedScanRegressionTests(unittest.TestCase):
    def test_last_segment_of_recorded_scan_is_allowed_without_reinspection(self):
        state = execution_state(look_angle_deg=75)
        for row in LATEST_SCAN_TRACE[:4]:
            state.observe(row['view'])
            state.accept(Decision.model_validate(row['decision']))
        state.observe(LATEST_SCAN_TRACE[4]['view'])
        self.assertFalse(state.scan.coverage_complete())
        self.assertEqual(state.allowed_look_directions(), ['left', 'right'])
        # 原任务此处被重复规则拒绝并改选导航，现在继续补完这一圈。
        self.assertEqual(state.scan.look_options()['right']['repeat_of'], 'O1')
        before = state.snapshot()
        with self.assertRaisesRegex(ValueError, '复核'):
            state.resolve(Decision(action='look', direction='right', reason='补完本轮环视'))
        self.assertEqual(state.snapshot(), before)

    def test_old_narrow_route_proposals_cannot_skip_space_assessment(self):
        from navigation_agent.app.deployments.scene import validate_navigation_scene
        for row in LATEST_SCAN_TRACE[4:]:
            with self.subTest(step=row['view']['step']):
                decision = Decision.model_validate(row['decision'])
                with self.assertRaisesRegex(ValueError, 'space'):
                    validate_navigation_scene(decision, 'go2_office')
                # 人工对原图的狭窄判断用于校验回归，不声称模型已识别。
                decision.navigation_check.space = 'narrow'
                with self.assertRaisesRegex(ValueError, '明确看到目标'):
                    validate_navigation_scene(decision, 'go2_office')


# 2026-10-12 11:32 篮球任务的 O17–O20 往返回归。
# 从该次 events.jsonl 摘取原始位姿、决策及动作结果；省略与本回归无关的区域字段。
# 原图仅保留相对引用，测试不加载图片，也不使用运行目录或仿真真值。
STATIONARY_TRACE = json.loads(r'''[
  {
    "view": {
      "id": "P1",
      "step": 1,
      "heading_deg": 0.0,
      "observer_position": [
        6.5,
        13.8
      ],
      "image": "images/001.jpg"
    },
    "decision": {
      "action": "look",
      "reason": "当前正面未见目标，先快速查看左侧未看区域以获取更多信息或发现其他入口。",
      "direction": "left",
      "candidates": [
        {
          "id": "new_1",
          "kind": "open_passage",
          "description": "右前方通往隔壁房间的门洞",
          "evidence": "可见门框及内部白色家具，地面连通。",
          "expected_new_view": "进入隔壁房间，查看内部地面和家具。"
        }
      ],
      "scene": {
        "landmarks": [
          "左侧带台灯的木桌",
          "正前方白框窗户",
          "右前方通往隔壁房间的门洞"
        ],
        "visible_ground": "深色木地板，前方开阔，右侧有阴影区域",
        "occluded": [],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      }
    },
    "result": {
      "reason": "turned"
    }
  },
  {
    "view": {
      "id": "P6",
      "step": 17,
      "heading_deg": -40.37608604806646,
      "observer_position": [
        2.4408527765794417,
        10.463170777928434
      ],
      "image": "images/017.jpg"
    },
    "decision": {
      "action": "look",
      "reason": "当前 R1 房间主要地面及左侧植物旁区域已检查完毕且无目标，需向右观察以寻找通往隔壁房间的门洞 C1，继续探索。",
      "direction": "right",
      "previous_action_review": {
        "outcome": "progress",
        "view_change": "uncertain",
        "evidence": "导航成功到达黄色矮柜与植物旁，当前视野清晰覆盖了该区域地面，确认此处无篮球，解决了原局部问题。"
      },
      "candidates": [
        {
          "id": "C1",
          "kind": "open_passage",
          "description": "右前方通往隔壁房间的门洞",
          "evidence": "根据拓扑记录，R1 右侧存在通往隔壁房间的门洞 C1，当前 R1 主要地面已查完，需前往该入口继续搜索。",
          "expected_new_view": "回到 O1 视角或面向右侧，确认门洞位置并进入隔壁房间查看地面。"
        }
      ],
      "scene": {
        "landmarks": [
          "黄色扶手椅",
          "绿色盆栽植物",
          "电视柜与显示器"
        ],
        "visible_ground": "深红褐色地板，黄色扶手椅前方及左侧保险柜旁地面可见，未发现篮球",
        "occluded": [],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      }
    },
    "result": {
      "reason": "turned"
    }
  },
  {
    "view": {
      "id": "P6",
      "step": 18,
      "heading_deg": -109.3602045818176,
      "observer_position": [
        2.4408527765794417,
        10.463170777928434
      ],
      "image": "images/018.jpg"
    },
    "decision": {
      "action": "orient",
      "reason": "当前 R1 房间主要地面已查完且无目标，需转回 O17 方向以接近未查入口 C13，继续探索隔壁房间。",
      "view_id": "O17",
      "observation_intent": {
        "reference_view_id": "O17",
        "question": "确认右前方通往隔壁房间的门洞 C13 的具体位置",
        "new_evidence": "当前 R1 房间主要地面及左侧植物旁区域已检查完毕且无目标，需转回 O17 方向以接近未查入口 C13。"
      },
      "candidates": [
        {
          "id": "C13",
          "kind": "open_passage",
          "description": "右前方通往隔壁房间的门洞",
          "evidence": "根据拓扑记录及历史图 O1/O17，R1 右侧存在通往隔壁房间的门洞，当前 R1 主要地面已查完，需前往该入口继续搜索。",
          "expected_new_view": "转回 O17 视角后，确认门洞位置并进入隔壁房间查看地面。"
        }
      ],
      "scene": {
        "landmarks": [
          "绿色盆栽植物",
          "木桌与台灯",
          "灰色椅子"
        ],
        "visible_ground": "深红褐色地板，木桌下方及植物旁地面可见，未发现篮球",
        "occluded": [],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      }
    },
    "result": {
      "reason": "oriented"
    }
  },
  {
    "view": {
      "id": "P6",
      "step": 19,
      "heading_deg": -40.605269166118724,
      "observer_position": [
        2.4408527765794417,
        10.463170777928434
      ],
      "image": "images/019.jpg"
    },
    "decision": {
      "action": "orient",
      "reason": "上次导航因引用历史图被拒绝，需先转回 O18 方向，利用新图确认 C13 入口位置后再进行导航。",
      "view_id": "O18",
      "previous_action_review": {
        "outcome": "uncertain",
        "view_change": "uncertain",
        "evidence": "上次动作未执行，当前仍停留在 O19 视角，需重新规划路径确认步骤。"
      },
      "observation_intent": {
        "reference_view_id": "O18",
        "question": "确认通往隔壁房间的门洞 C13 的具体位置和通行条件",
        "new_evidence": "当前 O19 视角主要看到电视柜和扶手椅，需转回 O18 方向并结合向右观察来定位门洞入口"
      },
      "candidates": [
        {
          "id": "C13",
          "kind": "open_passage",
          "description": "右前方通往隔壁房间的门洞",
          "evidence": "根据拓扑记录，R1 右侧存在通往隔壁房间的门洞 C13，当前 R1 主要地面已查完，需前往该入口继续搜索。",
          "expected_new_view": "进入隔壁房间，查看内部地面是否有篮球。"
        }
      ],
      "scene": {
        "landmarks": [
          "黄色扶手椅",
          "带显示器的电视柜",
          "绿色盆栽植物"
        ],
        "visible_ground": "深红褐色地板，黄色扶手椅前方及电视柜旁地面可见，未发现篮球",
        "occluded": [],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      }
    },
    "result": {
      "reason": "oriented"
    }
  },
  {
    "view": {
      "id": "P6",
      "step": 20,
      "heading_deg": -107.75592275545131,
      "observer_position": [
        2.4408527765794417,
        10.463170777928434
      ],
      "image": "images/020.jpg"
    },
    "decision": {
      "action": "orient",
      "reason": "当前 R1 房间主要地面已检查完毕且无目标，需转回 O19 方向以接近未查入口 C13，继续探索隔壁房间。",
      "view_id": "O19",
      "observation_intent": {
        "reference_view_id": "O19",
        "question": "确认 C13 入口位置以便导航进入隔壁房间",
        "new_evidence": "当前 O20 面向墙壁和桌子，无法看到 C13 入口；需转回 O19 方向（曾见黄色扶手椅和电视柜区域）以重新定位入口。"
      },
      "candidates": [
        {
          "id": "C13",
          "kind": "open_passage",
          "description": "右前方通往隔壁房间的门洞",
          "evidence": "根据拓扑记录及 O17 历史图，R1 右侧存在通往隔壁房间的门洞 C13，当前 R1 主要地面已查完，需前往该入口继续搜索。",
          "expected_new_view": "转回 O19 方向后，确认门洞位置并进入隔壁房间查看地面。"
        }
      ],
      "scene": {
        "landmarks": [
          "绿色盆栽植物",
          "木桌与灰色椅子",
          "黄色扶手椅边缘"
        ],
        "visible_ground": "深红褐色地板，木桌下方及植物旁地面可见，未发现篮球",
        "occluded": [],
        "target_seen": "no",
        "place_label": "",
        "objects": []
      }
    },
    "result": {
      "reason": "oriented"
    }
  }
]''')

LEGACY_STATIONARY_SIGHTING = json.loads(r'''{
  "id": "C13",
  "place_id": "P6",
  "attempt_count": 0,
  "failed_attempts": 0,
  "status": "untried",
  "origin_candidate_id": "C1",
  "kind": "open_passage",
  "description": "右前方通往隔壁房间的门洞",
  "evidence": "根据拓扑记录，R1 右侧存在通往隔壁房间的门洞 C1，当前 R1 主要地面已查完，需前往该入口继续搜索。",
  "expected_new_view": "回到 O1 视角或面向右侧，确认门洞位置并进入隔壁房间查看地面。",
  "last_seen_step": 17,
  "view_id": "O17",
  "source": "model_visual_report"
}''')

REJECTED_STATIONARY_NAVIGATION = json.loads(r'''{
  "scene": {
    "target_seen": "no",
    "visible_ground": "深红褐色地板，黄色扶手椅前方及电视柜旁地面可见，未发现篮球",
    "landmarks": [
      "黄色扶手椅",
      "带显示器的电视柜",
      "绿色盆栽植物"
    ]
  },
  "spatial_update": {
    "region_id": "R1",
    "label": "深色地板房间，有木桌和窗户",
    "transition": "same",
    "evidence": "当前图中的黄色扶手椅、电视柜及深红褐色地板与历史参照 R1/O17 一致，确认为同一房间。",
    "search_status": "checked",
    "remaining": []
  },
  "candidates": [
    {
      "id": "C13",
      "kind": "open_passage",
      "description": "右前方通往隔壁房间的门洞",
      "evidence": "根据拓扑记录及历史图 O1/O17，R1 右侧存在通往隔壁房间的门洞，当前 R1 主要地面已查完，需前往该入口继续搜索。",
      "expected_new_view": "进入隔壁房间，查看内部地面是否有篮球。"
    }
  ],
  "skill": "search_navigation",
  "action": "navigate",
  "candidate_id": "C13",
  "instruction": "Turn right and walk to the doorway leading to the next room.",
  "reason": "当前 R1 房间主要地面及左侧植物旁区域已检查完毕且无目标，需前往未查入口 C13 进入隔壁房间继续寻找篮球。",
  "target_labels": [
    "basketball"
  ],
  "previous_action_review": {
    "outcome": "progress",
    "evidence": "成功转回至 O17 方向，当前视野清晰显示了黄色扶手椅和电视柜区域，与 C13 入口位置相符。"
  }
}''')


class RecordedStationaryLoopTests(unittest.TestCase):
    def replay_to_step20(self, *, legacy_sighting=False):
        state = execution_state(look_angle_deg=70)
        for entry in STATIONARY_TRACE[:-1]:
            view = entry['view']
            state.observe(copy.deepcopy(view))
            state.accept(Decision.model_validate(entry['decision']))
            if legacy_sighting and view['step'] == 17:
                # 模拟修复前已产生的错误记录，验证限额不依赖清洗候选才能生效。
                state.candidates.append(copy.deepcopy(LEGACY_STATIONARY_SIGHTING))
                state.scan.views[-1]['candidate_ids'].append('C13')
            decision = entry['decision']
            state.scan.record_result(decision['action'], entry['result'], view_id=decision.get('view_id'))
        state.observe(copy.deepcopy(STATIONARY_TRACE[-1]['view']))
        return state

    def test_original_tail_stops_before_step20_motion(self):
        for legacy in (False, True):
            with self.subTest(legacy_sighting=legacy):
                state = self.replay_to_step20(legacy_sighting=legacy)
                before = state.snapshot()
                with self.assertRaisesRegex(ValueError, '次数已用尽'):
                    state.accept(Decision.model_validate(STATIONARY_TRACE[-1]['decision']))
                self.assertEqual(state.snapshot(), before)
                self.assertEqual(len(state.scan.actions), 4)
                options = state.action_options()
                self.assertNotIn('O19', options['orient_view_ids'])
                self.assertNotIn('left', options['look_directions'])
                self.assertEqual(options['unseen_via_orient'], [])
                state.resolve(Decision(action='look', direction='right', reason='继续看尚未看的右侧'))

    def test_step17_history_does_not_create_a_current_sighting(self):
        state = self.replay_to_step20()
        self.assertEqual(len(state.candidates), 1)
        candidate = state.candidates[0]
        self.assertEqual((candidate['id'], candidate['view_id'], candidate['last_seen_step']),
                         ('C1', 'O1', 1))
        self.assertEqual(candidate['place_id'], STATIONARY_TRACE[0]['view']['id'])
        self.assertEqual(state.scan.views[1]['candidate_ids'], [])

    def test_failed_verification_removes_local_hint_but_preserves_history(self):
        state = self.replay_to_step20(legacy_sighting=True)
        confirmed_view = next(view for view in state.scan.views if view['id'] == 'O19')
        self.assertEqual(confirmed_view['unconfirmed_candidate_ids'], ['C13'])
        option = next(item for item in state.action_options()['orient_options'] if item['view_id'] == 'O19')
        self.assertEqual(option['candidate_ids'], [])
        self.assertIn(LEGACY_STATIONARY_SIGHTING, state.candidates)
        self.assertIn('C13', state.scan.views[1]['candidate_ids'])

    def test_look_and_orient_share_budget_despite_progress_and_new_wording(self):
        state = self.replay_to_step20(legacy_sighting=True)
        decision = Decision.model_validate(STATIONARY_TRACE[-1]['decision'])
        decision.previous_action_review = ActionReview(outcome='progress', evidence='模型再次声称已获得进展')
        decision.observation_intent.new_evidence = '换措辞重新确认'
        for action in ('orient', 'look'):
            proposal = decision.model_copy(deep=True)
            proposal.action = action
            proposal.direction = 'left' if action == 'look' else None
            with self.subTest(action=action), self.assertRaisesRegex(ValueError, '次数已用尽'):
                state.resolve(proposal)

    def test_history_navigation_error_does_not_order_another_turn(self):
        state = execution_state(look_angle_deg=70)
        for entry in STATIONARY_TRACE[:-1]:
            state.observe(copy.deepcopy(entry['view']))
            if entry['view']['step'] == 19:
                break
            state.accept(Decision.model_validate(entry['decision']))
        before = state.snapshot()
        with self.assertRaises(ValueError) as rejected:
            state.resolve(Decision.model_validate(REJECTED_STATIONARY_NAVIGATION))
        self.assertNotIn('先 orient', str(rejected.exception))
        self.assertIn('结束这次检查', str(rejected.exception))
        self.assertEqual(state.snapshot(), before)


class ObservationBoundaryTests(unittest.TestCase):
    @staticmethod
    def observe(state, step, heading, x=0):
        state.observe(dict(id='P1' if x == 0 else 'P2', step=step, heading_deg=heading,
                           observer_position=[x, 0], image=f'images/{step:03d}.jpg'))

    @staticmethod
    def doorway(candidate_id='new_1', source='current'):
        return dict(id=candidate_id, kind='open_passage', description='可通行门洞',
                    evidence='当前图可见门洞开口和连续地面', expected_new_view='查看门内地面',
                    evidence_source=source)

    @staticmethod
    def orient(view_id, **extra):
        return Decision(action='orient', view_id=view_id, reason='转回确认入口',
                        observation_intent=dict(reference_view_id=view_id,
                                                question='入口是否可通行', new_evidence='准备接近已选入口'),
                        **extra)

    def initial_state(self):
        state = execution_state(look_angle_deg=70)
        self.observe(state, 1, 0)
        state.accept(Decision(action='look', direction='left', reason='先看左侧',
                              candidates=[self.doorway()]))
        state.scan.record_result('look', {'reason': 'turned'})
        self.observe(state, 2, 70)
        return state

    def test_history_is_not_refreshed_by_look_or_orient_or_unselected_navigation(self):
        for action in ('look', 'orient', 'navigate'):
            for source in (None, 'history'):
                with self.subTest(action=action, source=source):
                    state = self.initial_state()
                    original = copy.deepcopy(state.candidates[0])
                    candidate = self.doorway('C1', source)
                    if action == 'orient':
                        proposal = self.orient('O1', candidates=[candidate])
                    elif action == 'look':
                        proposal = Decision(action='look', direction='left', reason='继续查看',
                                            candidates=[candidate])
                    else:
                        proposal = Decision(action='navigate', instruction='Walk to the visible doorway.',
                                            reason='选择当前可见的另一个门洞', candidate_id='new_2',
                                            candidates=[candidate, self.doorway('new_2')])
                    state.accept(proposal)
                    self.assertEqual(state.candidates[0], original)
                    self.assertNotIn('C1', state.scan.views[-1]['candidate_ids'])

    def test_explicit_current_cannot_override_historical_image_reference(self):
        state = self.initial_state()
        original = copy.deepcopy(state.candidates)
        for candidate_id in ('C1', 'new_1'):
            candidate = self.doorway(candidate_id)
            candidate['evidence'] = '历史图 O1 的门洞及通路'
            state.accept(Decision(action='look', direction='left', reason='继续看未查方向',
                                  candidates=[candidate]))
            self.assertEqual(state.candidates, original)
            with self.assertRaisesRegex(ValueError, '历史图'):
                state.resolve(Decision(action='navigate', instruction='Walk to the doorway.',
                                       reason='返回历史门洞', candidate_id=candidate_id,
                                       candidates=[candidate]))

    def test_first_return_and_currently_confirmed_navigation_remain_allowed(self):
        state = self.initial_state()
        state.accept(self.orient('O1', candidates=[]))
        state.scan.record_result('orient', {'reason': 'oriented'}, view_id='O1')
        self.observe(state, 3, 0)
        selected = state.accept(Decision(
            action='navigate', instruction='Walk through the visible doorway.', reason='当前已确认入口，返回通行',
            candidates=[self.doorway('C1')], candidate_id='C1'))
        self.assertEqual(selected, 'C1')
        self.assertEqual(state.candidates[0]['view_id'], 'O3')
        self.assertEqual(state.scan.views[-1]['unconfirmed_candidate_ids'], [])
        self.assertEqual(state.scan.direction_visits(state.scan.views[0]), 2)

    def test_current_candidates_on_orient_are_registered(self):
        state = self.initial_state()
        state.accept(self.orient('O1', candidates=[self.doorway('new_1')]))
        self.assertEqual(state.scan.views[-1]['candidate_ids'], ['C2'])
        self.assertEqual(state.candidates[1]['view_id'], 'O2')

    def test_repeated_frames_and_failed_turn_do_not_consume_reinspection(self):
        state = self.initial_state()
        state.accept(self.orient('O1', candidates=[]))
        state.scan.record_result('orient', {'reason': 'orientation_unavailable'}, view_id='O1')
        for step in range(3, 20):
            self.observe(state, step, 70 + (step % 2))
        self.assertEqual(state.scan.return_count(state.scan.views[0]), 0)
        state.resolve(self.orient('O1'))

    def test_leaving_and_returning_keeps_history_and_allows_new_local_inspection(self):
        state = self.initial_state()
        state.accept(self.orient('O1', candidates=[]))
        self.observe(state, 3, 0)
        self.observe(state, 4, 70)
        with self.assertRaisesRegex(ValueError, '次数已用尽'):
            state.resolve(self.orient('O3'))
        self.observe(state, 5, 70, x=2)
        self.observe(state, 6, 70, x=0)
        state.resolve(self.orient('O3'))
        self.assertEqual(state.candidates[0]['view_id'], 'O1')
        self.assertEqual(len(state.scan.views), 6)

    def test_short_action_summary_cannot_erase_direction_budget(self):
        state = self.initial_state()
        state.accept(self.orient('O1', candidates=[]))
        self.observe(state, 3, 0)
        self.observe(state, 4, 70)
        for _ in range(13):
            state.scan.record_result('look', {'reason': 'turned'})
        self.assertEqual(len(state.scan.actions), 12)
        with self.assertRaisesRegex(ValueError, '次数已用尽'):
            state.resolve(self.orient('O3'))

    def test_failed_confirmation_no_longer_blocks_existing_recovery(self):
        state = self.initial_state()
        for step, heading in enumerate((140, 210, 280, 330, 0), 3):
            self.observe(state, step, heading)
        state.last_navigation = dict(step=1, observer_position=[-2, 0],
                                     result={'reason': 'model_stop'})
        self.assertFalse(state.action_options()['recover_allowed'])
        state.accept(self.orient('O5', candidates=[]))
        self.assertEqual(state.scan.views[-1]['unconfirmed_candidate_ids'], ['C1'])
        self.assertTrue(state.action_options()['recover_allowed'])
        self.assertEqual(state.candidates[0]['view_id'], 'O1')
        state.resolve(Decision(action='recover', reason='周围已查无通路，沿已记录来路退回'))
