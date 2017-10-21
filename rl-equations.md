# Equations for RL. 강화학습 공식

### 보상함수
$$ R^a_s = E[R_{t+1} | S_t=s, A_t = a]$$

### 상태 변환 확률

$$P^a_ss' = P[S_{t+1} = s' | S_t = s, A_t = a]$$

### 벨만 방정식
$$ v_{k+1}(s) = \sum_{a \in A}\pi(a|s)(R_s^a + \gamma v_k(s')) $$

### 감가율

$$\gamma \in [0,1]$$

$$\gamma^{k-1}R_{t+k}$$

### 정책

$$\pi (a|s) = P[A_t = a | S_t = s]$$

### 감가율을 적용한 보상들의 합

$$ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \dots $$

### 계산 가능한 벨만 방정식

$$ v_\pi (s) = \sum_{a \in A} \pi (a|s)\biggl(R_{t+1} + \gamma \sum_{s' \in S}P^a_{ss'}v_\pi(s')\biggl) $$

### 상태 변환 확률이 1인 벨만 기대 방정식

$$ v_\pi(s) = \sum_{a \in A} \pi(a|s) (R_{t+1} + \gamma v_\pi (s')) $$

### 벨만 최적 방정식

$$ v_*(s) = max_aE[R_{t+1} + \gamma v_*(S_{t+1}) | S_t = s, A_t = a] $$

### 계산 가능한 벨만 최적 방정식

$$ v_{k+1} (s) = max_{a \in A} (S^a_s + \gamma v_k(s')) $$

### 큐함수 정의

$$ q_\pi(s,a) = E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s, A_t = a] $$

### 반환값의 평균으로 가치함수를 추정

$$ v_\pi (s) \sim \frac1{N(s)} \sum^{N(s)}_{i=1}G_i(s) $$

### 몬테카를로 예측의 가치함수 업데이트 식

$$ V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t)) $$

### 반환값에 대한 기댓값으로 정의하는 가치함수

$$ v_\pi(s) = E_\pi [G_t | S_t = s] $$

$$ v_\pi(s) = E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s] $$

$$  R + \gamma V(S_{t+1}) = 업데이트의 목표$$

$$  \alpha(R + \gamma V(S_{t+1}) - V(S_t)) = 업데이트의 크기$$

### 살사의 큐함수 업데이트 식

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t))$$


$$ Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma Q(s', a') - Q(s,a)) $$

### 큐러닝을 통한 큐함수의 업데이트

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1} + \gamma \underset{a'}{\mathrm{max}}Q(S_{t+1}, a') - Q(S_t, A_t))$$

### 큐함수에 대한 벨만 최적 방정식

$$ q_*(s,a) = E[R_{t+1} + \gamma\underset{a'}{\mathrm{max}}q_*(S_{t+1}, a') | S_t = s, A_t = a] $$

### 큐러닝에서 큐함수의 업데이트 식

$$ Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1} + \gamma \underset{a'}{\mathrm{max}}Q(S_{t+1}, a') -Q(S_t, A_t)) $$

### A3C Policy Gradient 수식 유도

* Policy gradient의 정책 업데이트식 > 핵심은 Policy gradient

$$ \theta \leftarrow \theta + \alpha \nabla_\theta J(\theta) $$