import numpy as np

class ANFIS:
    """
    Adaptive Neuro-Fuzzy Inference System (ANFIS) for estimating VO2 consumption and energy expenditure.

    This class implements an ANFIS model to estimate oxygen consumption (VO2) based on operator
    physiological characteristics and heart rate. It uses fuzzy inference rules with Gaussian
    membership functions to model the relationship between operator attributes (age, weight, height,
    resting heart rate) and VO2 consumption at different heart rate levels.
    """
    
    def __init__(self, age: float, weight: float, height: float, hr_rest: float) -> None:
        """
        Initialize the ANFIS model with operator physiological characteristics.

        This constructor initializes the model with operator attributes, calculates BMI,
        and automatically estimates the three key parameters (vo2_rest, flex_point, slope,
        and intercept) required for VO2 estimation.
        """
        self.age = age
        self.weight = weight
        self.height = height
        self.hr_rest = hr_rest
        self.bmi = weight / ((height / 100) ** 2)
        
        self.vo2_rest = self.estimate_vo2_rest()
        self.flex_point = self.estimate_flex_point()
        self.slope, self.intercept = self.estimate_slope_intercept()
    
    def gaussmf(self, x: float, mean: float, sigma: float) -> float:
        """
        Calculate Gaussian membership function value.

        Computes the membership degree of a value `x` in a Gaussian fuzzy set characterized
        by a mean (center) and sigma (spread/width) parameter.
        """
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
    
    def weighted_sum(self, weights: list[float], outputs: list[float]) -> float:
        """
        Calculate weighted average of outputs using normalized weights.

        Computes a weighted sum where weights are normalized (sum to 1) before applying.
        This is used in fuzzy inference to combine outputs from multiple rules.
        """
        weights = np.array(weights)
        outputs = np.array(outputs)
        return np.dot(weights, outputs) / weights.sum() if weights.sum() != 0 else 0
    
    def estimate_vo2_rest(self) -> float:
        """
        Estimate VO2 consumption at rest using ANFIS fuzzy inference.

        Estimates the baseline oxygen consumption (VO2) at rest based on operator's
        age, weight, height, and resting heart rate. Uses two fuzzy inference rules
        with Gaussian membership functions and linear consequent functions.
        """
        rules = [
            {'age': (45.987, 7.051), 'weight': (78.873, 8.907), 'height': (172.98, 5.772), 'hr_rest': (64.131, 6.618),
             'coef': (-0.0857, 0.0062, -0.0444, 0.0186, 13.42)},
            {'age': (57.942, 7.167), 'weight': (77.666, 9.175), 'height': (175.23, 5.146), 'hr_rest': (60.628, 7.257),
             'coef': (-0.1122, 0.0137, -0.0826, 0.06263, 23.57)}
        ]
        
        results, weights = [], []
        for rule in rules:
            w = self.gaussmf(self.age, *rule['age']) * self.gaussmf(self.weight, *rule['weight']) * \
                self.gaussmf(self.height, *rule['height']) * self.gaussmf(self.hr_rest, *rule['hr_rest'])
            weights.append(w)
            a, b, c, e, f = rule['coef']
            results.append(a * self.age + b * self.weight + c * self.height + e * self.hr_rest + f)

        value = self.weighted_sum(weights, results)
        if value < 2.9:
            return 2.9
        elif value > 8:
            return 8
        else:
            return value
    
    def estimate_flex_point(self) -> float:
        """
        Estimate the flex point (heart rate threshold) using ANFIS fuzzy inference.

        Estimates the heart rate threshold that separates resting VO2 calculation from
        active VO2 calculation. Below this threshold, VO2 is constant (vo2_rest). Above
        this threshold, VO2 increases linearly with heart rate.
        """
        rules = [
            {'weight': (70.761, 5.393), 'bmi': (23.648, 1.436), 'hr_rest': (85, 4.137), 'coef': (0.4948, -0.6262, 1.7211, -71.225)},
            {'weight': (77.61, 5.393), 'bmi': (25.264, 1.436), 'hr_rest': (60.5, 4.137), 'coef': (0.5310, 0.6457, 1.0255, -41.181)},
            {'weight': (95.709, 5.393), 'bmi': (28.614, 1.436), 'hr_rest': (68.5, 4.137), 'coef': (-0.3444, 0.4635, 0.8339, 44.001)},
            {'weight': (68.04, 5.393), 'bmi': (24.135, 1.436), 'hr_rest': (76, 4.137), 'coef': (-0.0781, -1.9512, 1.3141, 37.66)},
            {'weight': (81.602, 5.393), 'bmi': (27.351, 1.436), 'hr_rest': (65.222, 4.137), 'coef': (-0.3544, 0.7387, 1.5339, -19.538)}
        ]
        
        results, weights = [], []
        for rule in rules:
            w = self.gaussmf(self.weight, *rule['weight']) * self.gaussmf(self.bmi, *rule['bmi']) * \
                self.gaussmf(self.hr_rest, *rule['hr_rest'])
            weights.append(w)
            b, d, e, f = rule['coef']
            results.append(b * self.weight + d * self.bmi + e * self.hr_rest + f)

        value = self.weighted_sum(weights, results)
        if value < 59:
            return 59
        elif value > 110:
            return 110
        else:
            return value
    
    def estimate_slope_intercept(self) -> tuple[float, float]:
        """
        Estimate slope and intercept parameters for active VO2 calculation using ANFIS fuzzy inference.

        Estimates the linear regression parameters (slope and intercept) used to calculate
        VO2 consumption when heart rate is above the flex point. The relationship is:
        VO2 = intercept + slope * heart_rate.
        """
        rules = [
            {
                'age': (42, 4.35), 'weight': (73.85, 5.393), 'height': (177, 3.29),
                'bmi': (23.56, 1.44), 'hr_rest': (67.75, 4.14),
                'slope_coef': (0.0001, 0.0129, -0.0050, -0.0425, -0.0006, 1.329),
                'intercept_coef': (-0.0741, -2.193, 1.33, 6.547, -0.2555, -225.3)
            },
            {
                'age': (43, 4.35), 'weight': (78.02, 5.39), 'height': (177.8, 3.29),
                'bmi': (24.68, 1.44), 'hr_rest': (64.08, 4.14),
                'slope_coef': (-0.0050, 0.0105, -0.0183, -0.0563, 0.0033, 4.208),
                'intercept_coef': (0.4359, 0.0667, 0.5215, 1.501, -0.5238, -139.3)
            }
        ]
        
        slope_results, intercept_results, weights = [], [], []
        for rule in rules:
            w = self.gaussmf(self.age, *rule['age']) * self.gaussmf(self.weight, *rule['weight']) * \
                self.gaussmf(self.height, *rule['height']) * self.gaussmf(self.bmi, *rule['bmi']) * \
                self.gaussmf(self.hr_rest, *rule['hr_rest'])
            weights.append(w)
            sa, sb, sc, sd, se, sf = rule['slope_coef']
            slope_results.append(sa * self.age + sb * self.weight + sc * self.height + sd * self.bmi + se * self.hr_rest + sf)
            ia, ib, ic, id, ie, i_f = rule['intercept_coef']
            intercept_results.append(ia * self.age + ib * self.weight + ic * self.height + id * self.bmi + ie * self.hr_rest + i_f)

        slope_val = self.weighted_sum(weights, slope_results)
        intercept_val = self.weighted_sum(weights, intercept_results)

        if slope_val < 0.1:
            slope_val = 0.1
        elif slope_val > 0.7:
            slope_val = 0.7

        if intercept_val < -35:
            intercept_val = -35
        elif intercept_val > -1:
            intercept_val = -1

        return slope_val, intercept_val

    def model(self, hr_current: float) -> float:
        """
        Estimate VO2 consumption based on current heart rate using a piecewise linear model.

        Calculates oxygen consumption (VO2) in ml/kg/min based on the current heart rate.
        Uses a piecewise function: constant VO2 at rest when heart rate is at or below the
        flex point, and linear increase when heart rate exceeds the flex point.
        """
        if hr_current <= self.flex_point:
            return self.vo2_rest
        else:
            return self.intercept + self.slope * hr_current
