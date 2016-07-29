------------------------------------
--- Kalman Filters and Smoothers ---
------------------------------------
{-
Written by:   Dominic Steinitz, Jacob West
Last updated: 2016-07-27

Summary: Linear and extended Kalman filters are provided, along with
their corresponding smoothers.
-}

---------------------------
--- File header pragmas ---
---------------------------


------------------------
--- Module / Exports ---
------------------------
module Numeric.Kalman
       (
         runKF, runEKF
       , runKS, runEKS
--         MultiNormal(..)                
       )
       where

---------------
--- Imports ---
---------------
import GHC.TypeLits
--import Numeric.Kalman.Internal
import Data.Random.Distribution.MultiNormal
import Numeric.LinearAlgebra.Static

----------------------
--- Kalman Filters ---
----------------------
{-

The Kalman filtering process is a recursive procedure as follows:

(1) Make a prediction of the current system state, given our previous
estimation of the system state.

(2) Update our prediction, given a newly acquired measurement.

The prediction and update steps depend on our system and measurement
models, as well as our current estimate of the system state.

-}

{-- runEKFPrediction --

Given our system model and our previous estimate of the system state,
we generate a prediction of the current system state by taking the
mean of our estimated state as our point estimate and evolving it one
step forward with the system evolution function.  In other words, we
predict that the current system state is the minimum mean squared
error (MMSE) state, which corresponds to the maximum a posteriori
(MAP) estimate, from the previous estimate evolved one step by the
system evolution function of our system model.

We also generate a predicted covariance matrix by updating the current
covariance and adding any noise estimate.  Updating the current
covariance requires the linearized version of the system evolution,
consequently the deviation of the actual system evolution from its
linearization is manifested in the newly predicted covariance.

Taken together, the prediction step generates a new Gaussian
distribution of predicted system states with mean the evolved MAP
state from the previous estimate.

-}

runEKFPrediction :: (KnownNat n) =>
                    (R n -> R n) ->      -- ^ System evolution function
                    (R n -> Sq n) ->     -- ^ Linearization of the system evolution at a point
                    Sym n ->             -- ^ Covariance matrix encoding system evolution noise
                    MultiNormal (R n) -> -- ^ Current estimate
                    MultiNormal (R n)    -- ^ New prediction

runEKFPrediction evolve linEvol sysCov estSys =
  MultiNormal predMu predCov
  where
    predMu  = evolve estMu
    predCov = sym $ lin <> estCov <> tr lin + (unSym sysCov)
    lin     = linEvol estMu

    estMu   = mu estSys
    estCov  = unSym . cov $ estSys


-- runKFPrediction :: (KnownNat n) =>
--                    (R n -> Sq n) ->     -- ^ Linear system evolution at a point
--                    Sym n ->             -- ^ Covariance matrix encoding system evolution noise
--                    MultiNormal (R n) -> -- ^ Current estimate
--                    MultiNormal (R n)    -- ^ New prediction

-- runKFPrediction linEvol sysCov estSys =
--   runEKFPrediction
--   (\sys -> (linEvol sys) #> sys)
--   linEvol
--   sysCov
--   estSys

{-- runEKFUpdate --

After a new measurement has been taken, we update our original
prediction of the current system state using the result of this
measurement.  This is accomplished by looking at how much the
measurement deviated from our prediction, and then updating our
estimated state accordingly.

This step requires the linearized measurement transformation.

-}
runEKFUpdate :: (KnownNat m, KnownNat n) =>
                (R n -> R m) ->      -- ^ System measurement function
                (R n -> L m n) ->    -- ^ Linearization of the measurement at a point
                Sym m ->             -- ^ Covariance matrix encoding measurement noise
                MultiNormal (R n) -> -- ^ Current prediction
                R m ->               -- ^ New measurement
                MultiNormal (R n)    -- ^ Updated prediction

runEKFUpdate measure linMeas measCov predSys newMeas =
  MultiNormal newMu newCov
  where
    newMu  = predMu + kkMat #> voff
    newCov = sym $ predCov - kkMat <> skMat <> tr kkMat

    predMu  = mu predSys
    predCov = unSym . cov $ predSys

    lin   = linMeas predMu
    voff  = newMeas - measure predMu
    skMat = lin <> predCov <> tr lin + (unSym measCov)
    kkMat = predCov <> tr lin <> (inv skMat)


-- runKFUpdate :: (KnownNat m, KnownNat n) =>
--                (R n -> L m n) ->    -- ^ Linear measurement operator at a point
--                Sym m ->             -- ^ Covariance matrix encoding measurement noise
--                MultiNormal (R n) -> -- ^ Current prediction
--                R m ->               -- ^ New measurement
--                MultiNormal (R n)    -- ^ Updated prediction

-- runKFUpdate linMeas measCov predSys newMeas =
--   runEKFUpdate
--   (\sys -> (linMeas sys #> sys))
--   linMeas
--   measCov
--   predSys
--   newMeas

{-- runEKF --

Here we combine the prediction and update setps applied to a new
measurement, thereby creating a single step of the (extended) Kalman
filter.

-}
runEKF :: (KnownNat m, KnownNat n) =>
          (R n -> R m) ->      -- ^ System measurement function
          (R n -> L m n) ->    -- ^ Linearization of the measurement at a point
          Sym m ->             -- ^ Covariance matrix encoding measurement noise
          (R n -> R n) ->      -- ^ System evolution function
          (R n -> Sq n) ->     -- ^ Linearization of the system evolution at a point
          Sym n ->             -- ^ Covariance matrix encoding system evolution noise
          MultiNormal (R n) -> -- ^ Current estimate
          R m ->               -- ^ New measurement
          MultiNormal (R n)    -- ^ New (filtered) estimate

runEKF measure linMeas measCov
  evolve linEvol sysCov
  estSys newMeas = updatedEstimate
  where
    predictedSystem = runEKFPrediction evolve linEvol sysCov estSys    
    updatedEstimate = runEKFUpdate measure linMeas measCov predictedSystem newMeas

runKF :: (KnownNat m, KnownNat n) =>
         (R n -> L m n) ->    -- ^ Linear measurement operator at a point
         Sym m ->             -- ^ Covariance matrix encoding measurement noise
         (R n -> Sq n) ->     -- ^ Linear system evolution at a point
         Sym n ->             -- ^ Covariance matrix encoding system evolution noise
         MultiNormal (R n) -> -- ^ Current estimate
         R m ->               -- ^ New measurement
         MultiNormal (R n)    -- ^ New (filtered) estimate

-- runKF linMeas measCov
--   linEvol sysCov
--   estSys newMeas = updatedEstimate
--   where
--     predictedSystem = runKFPrediction linEvol sysCov estSys    
--     updatedEstimate = runKFUpdate linMeas measCov predictedSystem newMeas

runKF linMeas measCov
  linEvol sysCov
  estSys newMeas =
  runEKF
  (\sys -> linMeas sys #> sys)
  linMeas measCov
  (\sys -> linEvol sys #> sys)
  linEvol sysCov
  estSys newMeas

------------------------
--- Kalman Smoothers ---
------------------------
{-

The Kalman smoothing process (sometimes also called the
Rauch-Tung-Striebel smoother or RTSS) is a recursive procedure for
improving previous estimates of the system state in light of more
recently obtained measurements.  That is, for a given state estimate
that was produced (by a Kalman filter) using only the available
historical data, we improve the estimate (smoothing it) using
information that only became available later, via additional
measurements after the estimate was originally made.  The result is an
estimate of the state of the system at that moment in its evolution,
taking full advantage of hindsight obtained from observations of the
system in the future of the current step.

Consequently, the recursive smoothing procedure progresses backwards
through the state evolution, beginning with the most recent
observation and updating past observations given the newer
information.  The update is made by measuring the deviation between
what the system actually did (via observation) and what our best
estimate of the system at that time predicted it would do, then
adjusting the current estimate in view of this deviation.

-}
runEKS :: (KnownNat n) =>
          (R n -> R n) ->      -- ^ System evolution function
          (R n -> Sq n) ->     -- ^ Linearization of the system evolution at a point
          Sym n ->             -- ^ Covariance matrix encoding system evolution noise
          MultiNormal (R n) -> -- ^ Future smoothed estimate
          MultiNormal (R n) -> -- ^ Present filtered estimate
          MultiNormal (R n)    -- ^ Present smoothed estimate

runEKS sysEvol linEvol sysCov future present =
  MultiNormal smMu smCov
  where
    smMu  = curMu + gkMat #> (futMu - predMu)
    smCov = sym $ curCov + gkMat <> (futCov - predCov) <> tr gkMat

    futMu  = mu future
    futCov = unSym . cov $ future

    curMu  = mu present
    curCov = unSym . cov $ present

    prevPred = runEKFPrediction sysEvol linEvol sysCov present
    predMu   = mu prevPred
    predCov  = unSym . cov $ prevPred

    gkMat = curCov <> tr lin <> inv predCov
    lin   = linEvol curMu


runKS :: (KnownNat n) =>
         (R n -> Sq n) ->     -- ^ Linear system evolution at a point
         Sym n ->             -- ^ Covariance matrix encoding system evolution noise
         MultiNormal (R n) -> -- ^ Future smoothed estimate
         MultiNormal (R n) -> -- ^ Present filtered estimate
         MultiNormal (R n)    -- ^ Present smoothed estimate

runKS linEvol sysCov future present =
  runEKS
  (\sys -> linEvol sys #> sys)
  linEvol
  sysCov
  future
  present
