# Intro to Reinforcement Learning

## Supervised Learning vs Unsupervised Learning vs Reinforcement Learning

### Supervised Learning
-  In supervised learning, labels are associated with the training data, and these labels are used to correct the model parameters. These labels give the model an idea of what is correct, and it tries to fix itself so that its predictions are more correct.

-  Supervised learning focuses on the attributes of each individual data point. These attributes are called features. Each data point is a list, or a vector, of such features, and this is called a feature vector. The input into the model is a feature vector, and the output is some kind of prediction, a classification or a regression. These feature vectors are typically called x variables. The outuput of the algorithm is a label. This a prediction your machine learning algorithm tries to make. 

-  There are two broad categories of labels in supervised learning. Categorical values are typically the output of any classification problem. Categorical values are discrete outputs, meaning they fall into a certain categories such as days of the weeks, months of the year, true or false, cat or dog, etc. The model can also predict values in a continuous range between some start and end value. This is the output of a regression model. For example, the price of a house in San Francisco, the price of Google stock next week, etc. These output values of the model are called the y varaibles.

-  Supervised learning makes the assumption that the input x, that is the feature vector or x variable, is linked to the output y by some function f, and the objective of the model is to learn what f is. This function f, which links the input to the output, is typically called mapping function.

### Unsupervised Learning
-  The unsupervised learning technique only have the underlying data to work with. There are no corresponding y variables or labels on this data. This technique involves setting up the model correctly so that it has the ability to learn structure or patterns in the underlying data. These algorithms are set up in such a manner such that they're able to self-discover the patterns and structure in the data without the help of any labeled instances.

-  One of the most common unsupervised learning algorithms is clustering. Clustering is used to find logical groups in the uderlying data. For example, when you cluster user data, you might want to find all users who like the same music in the same cluster. This will allow you to target specific ads to those users. There are many popular clustering algorithms out there. For example, K-means, mean shift clustering, hierarchical clustering, etc.

- Another common unsupervised learning technique is autoencoding for dimensionality reduction. Typically, when working with data sets, there is a huge nuber of features. But not all of these features might be significant, and some of these features might be correlated with each other. This is where dimensionality reduction can be used in order to find the latent factrs that drive the data. Principal component analysis, or PCA,  is a classic technique that is used for dimensionality reduction.

### Reinforcement Learning (RL)
-  Reinforcment learning is all about asking a number of different questions, that is exploring, and then figuring out which of the answers are correct, and then training an algorithm to make decisions based on these right answers.   

-  Neither supervised nor unsupervised learning will work in an unknown environment, where x is unknown and when that type of data has never been encountered before by that machine learning model. This unknown environment is where reinforcement learning operates. 

-  Reinforcement learning trains decision makers to take actions to maximize rewards in an uncertain environment. 
    -  Decision makers are a principal concept in a reinforcemnt learning algorithm. Decision makers are essentially software programs which take a series of actions based on decisions. Decision makers are also often called agents. So instead of a machine learning model or model paramters, there are reinforcement learning agents. The agent can be a car that's learning to self-drive or a programmatic model that's learning to trade. 

    -  The actions in RL refer to the decisions that the agent has taken. The  output of RL is a set of actions: do this, go left, go right, jump higher, rather than a set of predictions. All of these actions are geared towards a desirable outcome or an end. How long can you self driving car go without an accident? How can you maximize your returns from trading? These are our desired outcomes. These actions are determined using a reinforcement learning algorithm, and this algorithm is called the agent's policy. The agent uses this policy in order to take actions or make decisions. A self-driving car might have a policy if a human is withinn 5 meters of me, hit the brakes. An action for the trading algorithm might be if the market falls below a certain level, buy. All of these actions the agent takes have an objective. 

    -  The actions of the agent must be optimized to earn rewards or to avoid punishment, so there is positive reinforcement or negative reinforcement. The stock trading agent is rewarded whenever it makes money, and it's punished when it loses money. 

    -  All of these RL agents operate in some kind of external environment, and it is this external environment that imposes these rewards and punishments. Let's say the agent is a robot that's learning to walk. The external environment is the plane or the surface on which it's walking. If the agent is a thermostat, the external environment is the temperature or the electricity bill, humans fiddling with the temperature knob, etc. 

    -  An important part of reinforcement learning is the fact that the environment is uncertain or unknown. It's very complex. If the robot is learning to climb a hill, it has no idea of what pitfalls it might find. What if the robot uses a tree branch to swing itself up? Will the branch break, or will it be able to get higher up on the hill? Unless it's encoutered this particular tree and branch before, it has absolutely no idea, and this is where the training for the agent comes in. 
    
    -  The decision maker, or program, needs to be trained to explore this uncertain environment. Taking various routes to see which is the best way to get up this hill. Is it even possible? The robot needs to combine both caution and courage. It needs to explore new paths so that it can find new better ways, but it also might need to stick to familiar paths at some point in order to ensure it makes progress towards its goal. An overly adventurous agent might learn more about its environment, but it's also likely to fail fairly often. An overly cautious agent might learn very little about its environment. It might find one way up the hill, but it's possible that there are other better ways which it hasn't explored. RL agents have to maintain a tricky balance in order to maximize their rewards.
 
## Understanding the Reinforcement Learning Problem

### Modeling the Environment as a Markov Decision Process (MDP)
-  Markov Property: Future is independent of the past, given the present.

    -  Everything that we needed to learn from the past is embedded in the present, and there is no reason for us to look backwards. We have the information in the present state, and this present state can be used to model the future. 

-  At each time step...

    -  ...environment is in some state $S_{t}$

        -  If you try to imagine the state in the real world, let's say for a walking robot, a simplified representation of state will simply be the coordinates of the robot. The state can also contain information such as whether there's a hill up ahead, a hole to the left, etc.

    -  Decision maker can choose an action a

        -  The agent then observes the current state and then takes some action a within this state. What this action is depends on the policy theat the agent uses, and this policy has been predetermined when the agent explored the environment earlier.

    -  Moves environment to new state $S_{t+1}$

        -  The consequence of the agent taking this action a is that the state has now moved to a new state, $S_{t+1}$. This new state will have an entirely new set of oberservations and conditions.

    -  Decision maker receives reward $R_{a}$($S_{t}$, $S_{t+1}$)

        -  The decision maker then receives some reinforcement for this, some positive or negative reinforcemnt. Let's assume positive reinforcement for now. The decision maker recieves some reward for taking this action a when in state $S_{t}$.

        -  The reward depends only on the action that the decison maker took, the current state, and the next state. The reward is not of the form $R_{a}$($S_{t-2}$, $S_{t-1}$, $S_{t}$, $S_{t+1}$), which includes the previous state information. 

    -  $S_{t+1}$ depends only on a and $S_{t}$
    
        -  And this is the Markov property. The future state $S_{t+1}$ is completely independent of the past. $S_{t-1}$, $S_{t-2}$ are all irrevelant given that we know the present $S_{t}$. The path taken to get to the current state does not matter.

-  We need to model the environment so that the agent is able to explore it, finding the best possible action at every step. The MDP greatly simplifies the exploration of the environment because it allows the use of dynamic programming techniques, which makes policy search tractable. Finding the best policy to use in order to make decisions in a comple environment is now computable.

### Policy Search
-  Once you model environment, the next step is to perform a policy search algorithm to find the best policy for your agent to make decisions in this environment. 

-  Basic Elements of Reinforcement Learning (Recap):  

    -  Reward: Favorable result awarded for good actions. (The corollary is the negative awards, or the punishments, for bad actions.)

    -  Decision Maker: Software prgram that is competing for reward and trying to avoid punishments. 

    -  Policy: Algorithm to choose actions that will result in reward and avoid punishments. Policy is significant here because it tells the agent what needs to be done. Policy determines action.

-  The policy is a map or guide for the decision maker. Once the decision maker has a policy, he can use it to figure out what action to take. Each of these actions can be associated with an reward. the magnitude of the reward will depend on the action and the current state of the environemnt. These rewards can be negative as well. 

-  The agent's exploration of the environment determines the policy. It is the environment that tells you what actions under what conditions are good or bad. 

    -  Environment rewards some actions, punishes others

    -  Environments are uncertain, not known in advance

    -  Decision maker observes environment

    -  Decision maker learns to modify behavior accordingly

-  Example: Let's say you're potty training your puppy. When the puppy poops outside, you give the puppy a treat. When the puppy poops inside, you give the puppy a scolding. 

-  Policy Search:

    -  Find the "policy" that decision maker should follow

    -  Policy: Function P that takes in current state S, and returns action a
    
        -  a = P(S)
  
    -  Policy P should maximize cumulative rewards

-   Cumalitive Rewards:

    -  Little point maximizing immediate reward

        -  In chess, if your next move gains a pawn, but ignores a threat to your king...

        -  ...Game won't last long

        -  Need to balance immediate and deferred reward

    -  At time t, reward for next move is $R_{a}$($S_{t}$, $S_{t+1}$)

        -  And the move after that is $R_{a}$($S_{t+1}$, $S_{t+2}$)

        -  Repeat over an infinite horizon

    -  To calculate cumulative reward between now and infinity

        -  Add up all expected rewards 

        -  But discount future rewards by a discount factor γ

        -  0 < $\gamma$ <= 1

        -  Apply this discount factor successively to each future time

        -  Maximize $R_{a}(S_{t}, S_{t+1}) + \gamma R_{a}(S_{t+1}, S_{t+2}) + \gamma^{2}R_{a}(S_{t+2}, S_{t+3}) + \gamma^{3}R_{a}(S_{t+3}, S_{t+4}) + \gamma^{4}R_{a}(S_{t+4}, S_{t+5}) + ...$

            -  The rewards you might expect in future time periods are more uncertain, which is why you discount it. The further in the future a reward is, the more you discount it.
         
        -  Rewards depend on the action: a = P(S)

            -  Therefore, $R_{P(S)}(S_{t}, S_{t+1}) + \gamma R_{P(S)}(S_{t+1}, S_{t+2}) + \gamma^{2}R_{P(S)}(S_{t+2}, S_{t+3}) + \gamma^{3}R_{P(S)}(S_{t+3}, S_{t+4}) + \gamma^{4}R_{P(S)}(S_{t+4}, S_{t+5}) + ...$

            -  or $\sum_{t=0}^\infty \gamma^{t}R_{P(S)}(S_{t}, S_{t+1})$

                -  Find P to maximize this mathematical formula, which allows us to estimate the best possible policy for a particular environment.

                -  The process of solving this optimization problem is called Policy Search. We want to find the best policy to drive actions and maximize cumulative rewards.
             
## Watch videos 1-4 in this playlist: 
