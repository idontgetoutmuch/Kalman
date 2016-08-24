-- | The model for the extended Kalman filter is given by
--
-- \[
-- \begin{aligned}
-- \boldsymbol{x}_i &= \boldsymbol{a}(\boldsymbol{x}_{i-1}) + \boldsymbol{\psi}_{i-1} \\
-- \boldsymbol{y}_i &= {H}\boldsymbol{x}_i + \boldsymbol{\upsilon}_i
-- \end{aligned}
-- \]
--
-- where
--
-- * \(\boldsymbol{a}\)\ is some non-linear vector-valued state update
-- function.
--
-- * \(\boldsymbol{\psi}_{i}\) are independent identically normally
-- distributed random variables with mean 0 representing the fact that
-- the state update is noisy: \(\boldsymbol{\psi}_{i} \sim {\cal{N}}(0,Q)\).
--
-- * \({H}\) is a matrix which represents how we observe the
-- hidden state process.
--
-- * \(\boldsymbol{\upsilon}_i\) are independent identically normally
-- distributed random variables with mean 0 represent the fact that
-- the observations are noisy: \(\boldsymbol{\upsilon}_{i} \sim {\cal{N}}(0,R)\).
--
-- We assume the whole process starts at 0 with our belief of the state (aka the
-- prior state) being given by
-- \(\boldsymbol{x}_0 \sim {\cal{N}}(\boldsymbol{\mu}_0, \Sigma_0)\)
--
-- The Kalman filtering process is a recursive procedure as follows:
--
-- (1) Make a prediction of the current system state, given our previous
-- estimation of the system state.
--
-- (2) Update our prediction, given a newly acquired measurement.
--
-- The prediction and update steps depend on our system and measurement
-- models, as well as our current estimate of the system state.
--
-- @
--       kfiltered = scanl (runKF (const (const measMat)) (const measCovariance) (const evolMat) (const sysCovariance) 1) initEst measurements
-- @
module Numeric.Kalman
       (
         runKF, runKFPrediction, runKFUpdate
       , runEKF, runEKFPrediction, runEKFUpdate
       , runUKF, runUKFPrediction, runUKFUpdate
       , runKS, runEKS, runUKS
       )
       where

import GHC.TypeLits
import Data.Random.Distribution.MultiNormal
import Numeric.LinearAlgebra.Static

-- | Given our system model and our previous estimate of the system state,
-- we generate a prediction of the current system state by taking the
-- mean of our estimated state as our point estimate and evolving it one
-- step forward with the system evolution function.  In other words, we
-- predict that the current system state is the minimum mean squared
-- error (MMSE) state, which corresponds to the maximum a posteriori
-- (MAP) estimate, from the previous estimate evolved one step by the
-- system evolution function of our system model.
--
-- We also generate a predicted covariance matrix by updating the current
-- covariance and adding any noise estimate.  Updating the current
-- covariance requires the linearized version of the system evolution,
-- consequently the deviation of the actual system evolution from its
-- linearization is manifested in the newly predicted covariance.
--
-- Taken together, the prediction step generates a new Gaussian
-- distribution of predicted system states with mean the evolved MAP
-- state from the previous estimate.
runEKFPrediction
  :: (KnownNat n)
  => (a -> R n -> R n)  -- ^ System evolution function
  -> (a -> R n -> Sq n) -- ^ Linearization of the system evolution at a point
  -> (a -> Sym n)       -- ^ Covariance matrix encoding system evolution noise
  -> a                  -- ^ Dynamical input
  -> MultiNormal (R n)  -- ^ Current estimate
  -> MultiNormal (R n)  -- ^ New prediction

runEKFPrediction evolve linEvol sysCov input estSys =
  MultiNormal predMu predCov
  where
    predMu  = evolve input estMu
    predCov = sym $ lin <> estCov <> tr lin + unSym (sysCov input)

    estMu   = mu estSys
    estCov  = unSym . cov $ estSys
    lin     = linEvol input estMu

runKFPrediction
  :: (KnownNat n)
  => (a -> R n -> Sq n) -- ^ Linear system evolution at a point
  -> (a -> Sym n)       -- ^ Covariance matrix encoding system evolution noise
  -> a                  -- ^ Dynamical input
  -> MultiNormal (R n)  -- ^ Current estimate
  -> MultiNormal (R n)  -- ^ New prediction
runKFPrediction linEvol =
  runEKFPrediction (\inp sys -> linEvol inp sys #> sys) linEvol

-- runKFPrediction linEvol sysCov input estSys =
--   runEKFPrediction
--   (\inp sys -> (linEvol inp sys) #> sys)
--   linEvol
--   sysCov
--   input
--   estSys

-- | After a new measurement has been taken, we update our original
-- prediction of the current system state using the result of this
-- measurement.  This is accomplished by looking at how much the
-- measurement deviated from our prediction, and then updating our
-- estimated state accordingly.
--
-- This step requires the linearized measurement transformation.
runEKFUpdate
  :: (KnownNat m, KnownNat n)
  => (a -> R n -> R m)   -- ^ System measurement function
  -> (a -> R n -> L m n) -- ^ Linearization of the measurement at a point
  -> (a -> Sym m)        -- ^ Covariance matrix encoding measurement noise
  -> a                   -- ^ Dynamical input
  -> MultiNormal (R n)   -- ^ Current prediction
  -> R m                 -- ^ New measurement
  -> MultiNormal (R n)   -- ^ Updated prediction

runEKFUpdate measure linMeas measCov input predSys newMeas =
  MultiNormal newMu newCov
  where
    newMu  = predMu + kkMat #> voff
    newCov = sym $ predCov - kkMat <> skMat <> tr kkMat

    predMu  = mu predSys
    predCov = unSym . cov $ predSys

    lin   = linMeas input predMu
    voff  = newMeas - measure input predMu
    skMat = lin <> predCov <> tr lin + unSym (measCov input)
    kkMat = predCov <> tr lin <> inv skMat


runKFUpdate
  :: (KnownNat m, KnownNat n)
  => (a -> R n -> L m n) -- ^ Linear measurement operator at a point
  -> (a -> Sym m)        -- ^ Covariance matrix encoding measurement noise
  -> a                   -- ^ Dynamical input
  -> MultiNormal (R n)   -- ^ Current prediction
  -> R m                 -- ^ New measurement
  -> MultiNormal (R n)   -- ^ Updated prediction
runKFUpdate linMeas =
  runEKFUpdate (\inp sys -> linMeas inp sys #> sys) linMeas

-- runKFUpdate linMeas measCov input predSys newMeas =
--   runEKFUpdate
--   (\inp sys -> (linMeas inp sys #> sys))
--   linMeas
--   measCov
--   input
--   predSys
--   newMeas

-- | Here we combine the prediction and update setps applied to a new
-- measurement, thereby creating a single step of the (extended) Kalman
-- filter.
runEKF
  :: (KnownNat m, KnownNat n)
  => (a -> R n -> R m)   -- ^ System measurement function
  -> (a -> R n -> L m n) -- ^ Linearization of the measurement at a point
  -> (a -> Sym m)        -- ^ Covariance matrix encoding measurement noise
  -> (a -> R n -> R n)   -- ^ System evolution function
  -> (a -> R n -> Sq n)  -- ^ Linearization of the system evolution at a point
  -> (a -> Sym n)        -- ^ Covariance matrix encoding system evolution noise
  -> a                   -- ^ Dynamical input
  -> MultiNormal (R n)   -- ^ Current estimate
  -> R m                 -- ^ New measurement
  -> MultiNormal (R n)   -- ^ New (filtered) estimate

runEKF measure linMeas measCov
  evolve linEvol sysCov
  input estSys newMeas = updatedEstimate
  where
    predictedSystem = runEKFPrediction evolve linEvol sysCov input estSys
    updatedEstimate = runEKFUpdate measure linMeas measCov input predictedSystem newMeas

-- | The ordinary Kalman filter is a special case of the extended Kalman
-- filter, when the state update and measurement transformations are
-- already linear.
runKF
  :: (KnownNat m, KnownNat n)
  => (a -> R n -> L m n) -- ^ Linear measurement operator at a point
  -> (a -> Sym m)        -- ^ Covariance matrix encoding measurement noise
  -> (a -> R n -> Sq n)  -- ^ Linear system evolution at a point
  -> (a -> Sym n)        -- ^ Covariance matrix encoding system evolution noise
  -> a                   -- ^ Dynamical input
  -> MultiNormal (R n)   -- ^ Current estimate
  -> R m                 -- ^ New measurement
  -> MultiNormal (R n)   -- ^ New (filtered) estimate
runKF linMeas measCov
  linEvol sysCov
  input estSys newMeas = updatedEstimate
  where
    predictedSystem = runKFPrediction linEvol sysCov input estSys
    updatedEstimate = runKFUpdate linMeas measCov input predictedSystem newMeas

runUKFPrediction
  :: KnownNat n
  => (a -> R n -> R n) -- ^ System evolution function at a point
  -> (a -> Sym n)      -- ^ Covariance matrix encoding system evolution noise
  -> a                 -- ^ Dynamical input
  -> MultiNormal (R n) -- ^ Current estimate
  -> MultiNormal (R n) -- ^ Prediction

runUKFPrediction evolve sysCov input estSys =
  MultiNormal predMu predCov
  where
    predMu  = weightM0 * estMu' +
              sum (map (weightCM *) sigmaPoints')

    predCov = sym $ col (weightC0 * (estMu' - predMu)) <> row (estMu' - predMu) +
              sum (map (\sig ->
                          col (weightCM * (sig - predMu)) <> row (sig - predMu)
                       )
                   sigmaPoints') +
              unSym (sysCov input)

    estMu  = mu estSys
    estMu' = evolve input estMu

    sqCov  = chol $ cov estSys
    sqRows = map (* sqrt 3) $ toRows sqCov
    sigmaPoints = map (estMu +) sqRows ++ map (estMu -) sqRows -- 2n points
    sigmaPoints' = map (evolve input) sigmaPoints

    -- hand-tuned weights, more explanation required
    n = fromIntegral $ size estMu
    weightM0 = 1 - n / 3
    weightCM = 1 / 6
    weightC0 = 4 - n / 3 - 3 / n

runUKFUpdate
  :: (KnownNat n, KnownNat m)
  => (a -> R n -> R m) -- ^ Measurement transformation
  -> (a -> Sym m)      -- ^ Covariance matrix encoding measurement noise
  -> a                 -- ^ Dynamical input
  -> MultiNormal (R n) -- ^ Current prediction
  -> R m               -- ^ New measurement
  -> MultiNormal (R n) -- ^ Updated prediction

runUKFUpdate measure measCov input predSys newMeas =
  MultiNormal newMu newCov
  where
    newMu  = predMu + kkMat #> (newMeas - upMu)
    newCov = sym $ (unSym $ cov predSys) - kkMat <> skMat <> tr kkMat

    predMu  = mu predSys
    predMu' = measure input predMu

    kkMat = ckMat <> inv skMat
    upMu  = weightM0 * predMu' +
            sum (map (weightCM *) sigmaPoints')

    skMat = (unSym $ measCov input) +
            (col $ weightC0 * (predMu' - upMu)) <> (row $ predMu' - upMu) +
            sum (map (\sig ->
                        (col $ weightCM * (sig - upMu)) <> (row $ sig - upMu)
                     )
                 sigmaPoints')

    ckMat = sum $
            zipWith (\preds meas ->
                       (col $ weightCM' * (preds - predMu)) <> (row $ meas - upMu)
                    )
            sigmaPoints sigmaPoints'

    sqCov   = chol $ cov predSys
    sqRows = map (* sqrt 3) $ toRows sqCov
    sigmaPoints = map (predMu +) sqRows ++ map (predMu -) sqRows -- 2n points
    sigmaPoints' = map (measure input) sigmaPoints

    -- hand tuned weights, more exlanation required
    n = fromIntegral $ size predMu
    weightM0 = 1 - n / 3
    weightCM = 1 / 6
    weightCM' = 1 / 6
    weightC0 = 4 - n / 3 - 3 / n

runUKF
  :: (KnownNat m, KnownNat n)
  => (a -> R n -> R m) -- ^ System measurement function
  -> (a -> Sym m)      -- ^ Covariance matrix encoding measurement noise
  -> (a -> R n -> R n) -- ^ System evolution function
  -> (a -> Sym n)      -- ^ Covariance matrix encoding system evolution noise
  -> a                 -- ^ Dynamical input
  -> MultiNormal (R n) -- ^ Current estimate
  -> R m               -- ^ New measurement
  -> MultiNormal (R n) -- ^ New (filtered) estimate
runUKF measure measCov
  evolve sysCov
  input estSys newMeas = updatedEstimate
  where
    predictedSystem = runUKFPrediction evolve sysCov input estSys
    updatedEstimate = runUKFUpdate measure measCov input predictedSystem newMeas

-- | The Kalman smoothing process (sometimes also called the
-- Rauch-Tung-Striebel smoother or RTSS) is a recursive procedure for
-- improving previous estimates of the system state in light of more
-- recently obtained measurements.  That is, for a given state estimate
-- that was produced (by a Kalman filter) using only the available
-- historical data, we improve the estimate (smoothing it) using
-- information that only became available later, via additional
-- measurements after the estimate was originally made.  The result is an
-- estimate of the state of the system at that moment in its evolution,
-- taking full advantage of hindsight obtained from observations of the
-- system in the future of the current step.
--
-- Consequently, the recursive smoothing procedure progresses backwards
-- through the state evolution, beginning with the most recent
-- observation and updating past observations given the newer
-- information.  The update is made by measuring the deviation between
-- what the system actually did (via observation) and what our best
-- estimate of the system at that time predicted it would do, then
-- adjusting the current estimate in view of this deviation.
runEKS
  :: (KnownNat n)
  => (a -> R n -> R n)  -- ^ System evolution function
  -> (a -> R n -> Sq n) -- ^ Linearization of the system evolution at a point
  -> (a -> Sym n)       -- ^ Covariance matrix encoding system evolution noise
  -> a                  -- ^ Dynamical input
  -> MultiNormal (R n)  -- ^ Future smoothed estimate
  -> MultiNormal (R n)  -- ^ Present filtered estimate
  -> MultiNormal (R n)  -- ^ Present smoothed estimate

runEKS sysEvol linEvol sysCov input future present =
  MultiNormal smMu smCov
  where
    smMu  = curMu + gkMat #> (futMu - predMu)
    smCov = sym $ curCov + gkMat <> (futCov - predCov) <> tr gkMat

    futMu  = mu future
    futCov = unSym . cov $ future

    curMu  = mu present
    curCov = unSym . cov $ present

    prevPred = runEKFPrediction sysEvol linEvol sysCov input present
    predMu   = mu prevPred
    predCov  = unSym . cov $ prevPred

    gkMat = curCov <> tr lin <> inv predCov
    lin   = linEvol input curMu


runKS
  :: (KnownNat n)
  => (a -> R n -> Sq n) -- ^ Linear system evolution at a point
  -> (a -> Sym n)       -- ^ Covariance matrix encoding system evolution noise
  -> a                  -- ^ Dynamical input
  -> MultiNormal (R n)  -- ^ Future smoothed estimate
  -> MultiNormal (R n)  -- ^ Present filtered estimate
  -> MultiNormal (R n)  -- ^ Present smoothed estimate
runKS linEvol =
  runEKS (\inp sys -> linEvol inp sys #> sys) linEvol

-- runKS linEvol sysCov input future present =
--   runEKS
--   (\inp sys -> linEvol inp sys #> sys)
--   linEvol
--   sysCov
--   input
--   future
--   present

-- | Unscented Kalman smoother
runUKS
  :: (KnownNat n)
  => (a -> R n -> R n)  -- ^ System evolution function
  -> (a -> Sym n)       -- ^ Covariance matrix encoding system evolution noise
  -> a                  -- ^ Dynamical input
  -> MultiNormal (R n)  -- ^ Future smoothed estimate
  -> MultiNormal (R n)  -- ^ Present filtered estimate
  -> MultiNormal (R n)  -- ^ Present smoothed estimate
runUKS evolve sysCov input future present =
  MultiNormal smMu smCov
  where
    smMu  = curMu + gkMat #> (futMu - predMu)
    smCov = sym $ curCov + gkMat <> (futCov - predCov) <> tr gkMat

    futMu  = mu future
    futCov = unSym . cov $ future

    curMu  = mu present
    curCov = unSym . cov $ present

    prevPred = runUKFPrediction evolve sysCov input present
    predMu   = mu prevPred
    predCov  = unSym . cov $ prevPred

    gkMat = dkMat <> inv predCov

    dkMat = sum $
            zipWith (\pres fut ->
                       (col $ weightCM * (pres - curMu)) <> (row $ fut - predMu)
                    )
            sigmaPoints sigmaPoints'

    sqCov   = chol $ cov present
    sqRows = map (* sqrt 3) $ toRows sqCov
    sigmaPoints = map (curMu +) sqRows ++ map (curMu -) sqRows -- 2n points
    sigmaPoints' = map (evolve input) sigmaPoints

    -- hand tuned weights, more exlanation required
    weightCM = 1 / 6
