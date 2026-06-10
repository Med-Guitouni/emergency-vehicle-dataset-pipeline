import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
from collections import Counter

class SceneClassifier:
    """
    Classifies the driving scenario type per frame using MIT Places365 ResNet18(CNN ).
    Uses a consecutive frames confirmation approach instead of a rolling window:
    a new scenario is only accepted if the same prediction appears 3 times in a row.
    This prevents single noisy frames from flipping the scenario type while still
    detecting real transitions within 3 seconds.

    """

    SCENARIO_MAP = {
        'highway': 'highway',
        'forest_road': 'highway',
        'raceway': 'highway',
        'bridge': 'highway',
        'viaduct': 'highway',
        'intersection': 'intersection',
        'crossroads': 'intersection',
        'traffic_light': 'intersection',
        'roundabout': 'roundabout',
        'rotary': 'roundabout',
        'street': 'urban',
        'road': 'urban',
        'avenue': 'urban',
        'parking_lot': 'urban',
        'residential_neighborhood': 'urban'
    }

    def __init__(self, min_consecutive=3):
        """
        min_consecutive: number of consecutive identical predictions required
        to confirm a scenario change. Set to 3 to balance responsiveness
        and stability.
        """
        print("Loading Places365 scene classifier...")

        url = 'https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt'
        self.classes = []
        with urllib.request.urlopen(url) as f:
            for line in f:
                self.classes.append(line.decode().strip().split(' ')[0][3:])

        self.model = models.resnet18(num_classes=365)
        checkpoint = torch.hub.load_state_dict_from_url(
            'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar',
            map_location='cpu'
        )
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.min_consecutive = min_consecutive
        self.current_scenario = 'unknown'
        self.candidate_scenario = None
        self.consecutive_count = 0

        print("Scene classifier ready")

    def _predict(self, frame):
        """Run Places365 on one frame and return mapped scenario type"""
        img = Image.fromarray(frame[:, :, ::-1])
        input_tensor = self.transform(img).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            top5_probs, top5_idx = probs.topk(5)

        top5 = [(self.classes[top5_idx[0][i]], top5_probs[0][i].item())
                for i in range(5)]

        return self._map_to_scenario(top5)

    def _map_to_scenario(self, top5):
        """
        Maps Places365 top 5 predictions to our 4 scenario types
        using weighted probability voting across all top 5 predictions.
        If unknown wins fall back to second highest scoring scenario.
        """
        scenario_scores = {
            'highway': 0.0,
            'intersection': 0.0,
            'roundabout': 0.0,
            'urban': 0.0,
            'unknown': 0.0
        }

        for scene, prob in top5:
            scene_lower = scene.lower()
            matched = False
            for keyword, scenario in self.SCENARIO_MAP.items():
                if keyword in scene_lower:
                    scenario_scores[scenario] += prob
                    matched = True
                    break
            if not matched:
                scenario_scores['unknown'] += prob

        sorted_scores = sorted(scenario_scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_scores[0][0] == 'unknown':
            return sorted_scores[1][0]
        return sorted_scores[0][0]

    def classify(self, frame):
        """
        Classify scenario type with consecutive frames confirmation.
        Only accepts a new scenario after min_consecutive identical predictions.
        Returns the current confirmed scenario until a change is confirmed.
        """
        raw = self._predict(frame)

        if raw == self.current_scenario:
            # prediction matches current - reset any pending candidate
            self.candidate_scenario = None
            self.consecutive_count = 0
            return self.current_scenario

        if raw == self.candidate_scenario:
            # same candidate as before - increment counter
            self.consecutive_count += 1
        else:
            # new different prediction - start fresh candidate
            self.candidate_scenario = raw
            self.consecutive_count = 1

        # if candidate confirmed enough times - accept it
        if self.consecutive_count >= self.min_consecutive:
            self.current_scenario = self.candidate_scenario
            self.candidate_scenario = None
            self.consecutive_count = 0

        return self.current_scenario

    def reset(self):
        """Reset state between videos"""
        self.current_scenario = 'unknown'
        self.candidate_scenario = None
        self.consecutive_count = 0