-----------------------------------------------------------------------------
-- |
-- Module      :  Numeric.Kalman
-- Copyright   :  (c) 2016 FP Complete Corporation
-- License     :  MIT (see LICENSE)
-- Maintainer  :  dominic@steinitz.org
--
-- =The Theory
-- The model for the extended Kalman filter is given by
--
-- \[
-- \begin{aligned}
-- \boldsymbol{x}_i &= \boldsymbol{a}_i(\boldsymbol{x}_{i-1}) + \boldsymbol{\psi}_{i-1} \\
-- \boldsymbol{y}_i &= \boldsymbol{h}_i(\boldsymbol{x}_i) + \boldsymbol{\upsilon}_i
-- \end{aligned}
-- \]
--
-- where
--
-- * \(\boldsymbol{a_i}\)\ is some non-linear vector-valued possibly
-- time-varying state update function.
--
-- * \(\boldsymbol{\psi}_{i}\) are independent normally
-- distributed random variables with mean 0 representing the fact that
-- the state update is noisy: \(\boldsymbol{\psi}_{i} \sim {\cal{N}}(0,Q_i)\).
--
-- * \(\boldsymbol{h}_i\)\ is some non-linear vector-valued possibly time-varying
-- function describing how we observe the hidden state process.
--
-- * \(\boldsymbol{\upsilon}_i\) are independent normally
-- distributed random variables with mean 0 represent the fact that
-- the observations are noisy: \(\boldsymbol{\upsilon}_{i} \sim {\cal{N}}(0,R_i)\).
--
-- Note that in most presentations of the Kalman filter (the
-- [wikipedia](https://en.wikipedia.org/wiki/Kalman_filter)
-- presentation being an exception), the state update function and the
-- observation function are taken to be constant over time as is the
-- noise for the state and the noise for
-- the observation. In symbols, \(\forall i \, \boldsymbol{a}_i = \boldsymbol{a}\)
-- for some \(\boldsymbol{a}\), \(\forall i \, \boldsymbol{h}_i = \boldsymbol{h}\)
-- for some \(\boldsymbol{h}\),
-- \(\forall i \, Q_i = Q\) for some \(Q\) and \(\forall i \, \boldsymbol{a}_i = R\)
-- for some \(R\).
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
-- = An Extended Example
--
-- The equation of motion for a pendulum of unit length subject to
-- [Gaussian white
-- noise](https://en.wikipedia.org/wiki/White_noise#Mathematical_definitions)
-- is
--
-- \[
-- \frac{\mathrm{d}^2\alpha}{\mathrm{d}t^2} = -g\sin\alpha + w(t)
-- \]
--
-- We can discretize this via the usual [Euler method](https://en.wikipedia.org/wiki/Euler_method)
--
-- \[
-- \begin{bmatrix}
-- x_{1,i} \\
-- x_{2,i}
-- \end{bmatrix}
-- =
-- \begin{bmatrix}
-- x_{1,i-1} + x_{2,i-1}\Delta t \\
-- x_{2,i-1} - g\sin x_{1,i-1}\Delta t
-- \end{bmatrix}
-- +
-- \mathbf{q}_{i-1}
-- \]
--
-- where \(q_i \sim {\mathcal{N}}(0,Q)\) and
--
-- \[
-- Q
-- =
-- \begin{bmatrix}
-- \frac{q^c \Delta t^3}{3} & \frac{q^c \Delta t^2}{2} \\
-- \frac{q^c \Delta t^2}{2} & {q^c \Delta t}
-- \end{bmatrix}
-- \]
--
-- Assume that we can only measure the horizontal position of the
-- pendulum and further that this measurement is subject to error so that
--
-- \[
-- y_i = \sin x_i + r_i
-- \]
--
-- where \(r_i \sim {\mathcal{N}}(0,R)\).
--
-- First let's set the time step and the acceleration caused by earth's gravity.
--
-- > {-# LANGUAGE DataKinds #-}
-- >
-- > import Numeric.Kalman
-- >
-- > deltaT, g :: Double
-- > deltaT = 0.01
-- > g  = 9.81
--
-- And set up the noises for the filter.
--
-- > bigQ :: Sym 2
-- > bigQ = sym $ matrix bigQl
-- >
-- > qc :: Double
-- > qc = 0.01
-- >
-- > bigQl :: [Double]
-- > bigQl = [ qc * deltaT^3 / 3, qc * deltaT^2 / 2,
-- >           qc * deltaT^2 / 2,       qc * deltaT
-- >         ]
-- >
-- > bigR :: Sym 1
-- > bigR  = sym $ matrix [0.1]
--
-- The state update and observation functions:
--
-- > stateUpdate :: R 2 -> R 2
-- > stateUpdate u =  vector [x1 + x2 * deltaT, x2 - g * (sin x1) * deltaT]
-- >   where
-- >     (x1, w) = headTail u
-- >     (x2, _) = headTail w
-- >
-- > observe :: R 2 -> R 1
-- > observe a = vector [sin x] where x = fst $ headTail a
--
-- For the extended filter we need the derivatives of the state update
-- and observation functions. We create these by hand; another
-- possibility would be to use [automatic
-- differentiation](https://hackage.haskell.org/package/ad).
--
-- > linearizedObserve :: R 2 -> L 1 2
-- > linearizedObserve a = matrix [cos x, 0.0] where x = fst $ headTail a
-- >
-- > linearizedStateUpdate :: R 2 -> Sq 2
-- > linearizedStateUpdate u = matrix [1.0,                    deltaT,
-- >                                   -g * (cos x1) * deltaT,    1.0]
-- >   where
-- >     (x1, _) = headTail u
--
-- Now we can create extended and unscented filters which consume a
-- single observation.
--
-- > singleEKF = runEKF (const observe) (const linearizedObserve) (const bigR)
-- >              (const stateUpdate) (const linearizedStateUpdate) (const bigQ)
-- >              undefined
--
-- > singleUKF = runUKF (const observe) (const bigR) (const stateUpdate) (const bigQ)
-- >              undefined
--
-- We start off not too far from the actual value.
--
-- > initialDist =  (vector [1.6, 0.0],
-- >                 sym $ matrix [0.1, 0.0,
-- >                               0.0, 0.1])
--
-- Using some data generated using code made available with Simo
-- Särkkä's
-- [book](http://www.cambridge.org/gb/academic/subjects/statistics-probability/applied-probability-and-stochastic-networks/bayesian-filtering-and-smoothing),
-- we can track the pendulum using the extended Kalman filter.
--
-- > multiEKF obs = scanl singleEKF initialDist (map (vector . pure) obs)
--
-- And then plot the results.
--
-- <<diagrams/src_Numeric_Kalman_diagE.svg#diagram=diagE&height=600&width=500>>
--
-- And also track it using the unscented Kalman filter.
--
-- > multiUKF obs = scanl singleUKF initialDist (map (vector . pure) obs)
--
-- And also plot the results
--
-- <<diagrams/src_Numeric_Kalman_diagU.svg#diagram=diagU&height=600&width=500>>
--
-- === Code for Plotting
--
-- The full code for plotting the results:
--
-- > import qualified Graphics.Rendering.Chart as C
-- > import Graphics.Rendering.Chart.Backend.Diagrams
-- > import Data.Colour
-- > import Data.Colour.Names
-- > import Data.Default.Class
-- > import Control.Lens
-- >
-- > import Data.Csv
-- > import System.IO hiding ( hGetContents )
-- > import Data.ByteString.Lazy ( hGetContents )
-- > import qualified Data.Vector as V
-- >
-- > import Numeric.LinearAlgebra.Static
-- >
-- > chartEstimated :: String ->
-- >               [(Double, Double)] ->
-- >               [(Double, Double)] ->
-- >               [(Double, Double)] ->
-- >               C.Renderable ()
-- > chartEstimated title acts obs ests = C.toRenderable layout
-- >   where
-- >
-- >     actuals = C.plot_lines_values .~ [acts]
-- >             $ C.plot_lines_style  . C.line_color .~ opaque red
-- >             $ C.plot_lines_title .~ "Actual Trajectory"
-- >             $ C.plot_lines_style  . C.line_width .~ 1.0
-- >             $ def
-- >
-- >     measurements = C.plot_points_values .~ obs
-- >                  $ C.plot_points_style  . C.point_color .~ opaque blue
-- >                  $ C.plot_points_title .~ "Measurements"
-- >                  $ def
-- >
-- >     estimas = C.plot_lines_values .~ [ests]
-- >             $ C.plot_lines_style  . C.line_color .~ opaque black
-- >             $ C.plot_lines_title .~ "Inferred Trajectory"
-- >             $ C.plot_lines_style  . C.line_width .~ 1.0
-- >             $ def
-- >
-- >     layout = C.layout_title .~ title
-- >            $ C.layout_plots .~ [C.toPlot actuals, C.toPlot measurements, C.toPlot estimas]
-- >            $ C.layout_y_axis . C.laxis_title .~ "Angle / Horizontal Displacement"
-- >            $ C.layout_y_axis . C.laxis_override .~ C.axisGridHide
-- >            $ C.layout_x_axis . C.laxis_title .~ "Time"
-- >            $ C.layout_x_axis . C.laxis_override .~ C.axisGridHide
-- >            $ def
-- >
-- > diagE = do
-- >   h <- openFile "matlabRNGs.csv" ReadMode
-- >   cs <- hGetContents h
-- >   let df = (decode NoHeader cs) :: Either String (V.Vector (Double, Double))
-- >   case df of
-- >     Left _ -> error "Whatever"
-- >     Right generatedSamples -> do
-- >       let xs = take 500 (multiEKF $ V.toList $ V.map fst generatedSamples)
-- >       let mus = map (fst . headTail . fst) xs
-- >       let obs = V.toList $ V.map fst generatedSamples
-- >       let acts = V.toList $ V.map snd generatedSamples
-- >       denv <- defaultEnv C.vectorAlignmentFns 600 500
-- >       let charte = chartEstimated "Extended Kalman Filter"
-- >                                   (zip [0,1..] acts)
-- >                                   (zip [0,1..] obs)
-- >                                   (zip [0,1..] mus)
-- >       return $ fst $ runBackend denv (C.render charte (600, 500))
-- >
-- > diagU = do
-- >   h <- openFile "matlabRNGs.csv" ReadMode
-- >   cs <- hGetContents h
-- >   let df = (decode NoHeader cs) :: Either String (V.Vector (Double, Double))
-- >   case df of
-- >     Left _ -> error "Whatever"
-- >     Right generatedSamples -> do
-- >       let ys = take 500 (multiUKF $ V.toList $ V.map fst generatedSamples)
-- >       let nus = map (fst . headTail . fst) ys
-- >       let obs = V.toList $ V.map fst generatedSamples
-- >       let acts = V.toList $ V.map snd generatedSamples
-- >       denv <- defaultEnv C.vectorAlignmentFns 600 500
-- >       let charte = chartEstimated "Unscented Kalman Filter"
-- >                                   (zip [0,1..] acts)
-- >                                   (zip [0,1..] obs)
-- >                                   (zip [0,1..] nus)
-- >       return $ fst $ runBackend denv (C.render charte (600, 500))
--
-----------------------------------------------------------------------------

module Numeric.Kalman
       (
         runKF, runKFPrediction, runKFUpdate
       , runEKF, runEKFPrediction, runEKFUpdate
       , runUKF, runUKFPrediction, runUKFUpdate
       , runKS, runEKS, runUKS
       )
       where

import GHC.TypeLits
import Numeric.LinearAlgebra.Static
import Data.Maybe ( fromJust )

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
  :: KnownNat n
  => (a -> R n -> R n)  -- ^ System evolution function \(\boldsymbol{a}_n(s)\)
  -> (a -> R n -> Sq n) -- ^ Linearization of the system evolution at a point \(\frac{\partial \boldsymbol{a}_n}{\partial s}\big|_s\)
  -> (a -> Sym n)       -- ^ Covariance matrix encoding system evolution noise \(Q_n\)
  -> a                  -- ^ Dynamical input \(n\)
  -> (R n, Sym n)       -- ^ Current estimate \((\hat{\boldsymbol{x}}_{n-1}, \hat{\boldsymbol{\Sigma}}_{n-1})\)
  -> (R n, Sym n)       -- ^ New prediction \((\hat{\boldsymbol{x}}_n, \hat{\boldsymbol{\Sigma}}_n)\)

runEKFPrediction evolve linEvol sysCov input (estMu, estCov) =
  (predMu, predCov)
  where
    predMu  = evolve input estMu
    predCov = sym $ lin <> unSym estCov <> tr lin + unSym (sysCov input)
    lin     = linEvol input estMu

runKFPrediction
  :: (KnownNat n)
  => (a -> R n -> Sq n) -- ^ Linear system evolution at a point
  -> (a -> Sym n)       -- ^ Covariance matrix encoding system evolution noise
  -> a                  -- ^ Dynamical input
  -> (R n, Sym n)       -- ^ Current estimate
  -> (R n, Sym n)       -- ^ New prediction
runKFPrediction linEvol =
  runEKFPrediction (\inp sys -> linEvol inp sys #> sys) linEvol

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
  -> (R n, Sym n)        -- ^ Current prediction
  -> R m                 -- ^ New measurement
  -> (R n, Sym n)        -- ^ Updated prediction

runEKFUpdate measure linMeas measCov input (predMu, predCov') newMeas =
  (newMu, newCov)
  where
    newMu  = predMu + kkMat #> voff
    newCov = sym $ predCov - kkMat <> skMat <> tr kkMat

    predCov = unSym predCov'

    lin   = linMeas input predMu
    voff  = newMeas - measure input predMu
    skMat = lin <> predCov <> tr lin + unSym (measCov input)
    kkMat = predCov <> tr lin <> unsafeInv skMat


runKFUpdate
  :: (KnownNat m, KnownNat n)
  => (a -> R n -> L m n) -- ^ Linear measurement operator at a point
  -> (a -> Sym m)        -- ^ Covariance matrix encoding measurement noise
  -> a                   -- ^ Dynamical input
  -> (R n, Sym n)        -- ^ Current prediction
  -> R m                 -- ^ New measurement
  -> (R n, Sym n)        -- ^ Updated prediction
runKFUpdate linMeas =
  runEKFUpdate (\inp sys -> linMeas inp sys #> sys) linMeas

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
  -> (R n, Sym n)        -- ^ Current estimate
  -> R m                 -- ^ New measurement
  -> (R n, Sym n)        -- ^ New (filtered) estimate
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
  -> (R n, Sym n)        -- ^ Current estimate
  -> R m                 -- ^ New measurement
  -> (R n, Sym n)        -- ^ New (filtered) estimate
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
  -> (R n, Sym n)      -- ^ Current estimate
  -> (R n, Sym n)      -- ^ Prediction

runUKFPrediction evolve sysCov input (estMu, estCov) =
  (predMu, predCov)
  where
    predMu  = weightM0 * estMu' +
              sum (map (weightCM *) sigmaPoints')

    predCov = sym $ col (weightC0 * (estMu' - predMu)) <> row (estMu' - predMu) +
              sum (map (\sig ->
                          col (weightCM * (sig - predMu)) <> row (sig - predMu)
                       )
                   sigmaPoints') +
              unSym (sysCov input)

    estMu' = evolve input estMu

    sqCov  = chol estCov
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
  -> (R n, Sym n)      -- ^ Current prediction
  -> R m               -- ^ New measurement
  -> (R n, Sym n)      -- ^ Updated prediction

runUKFUpdate measure measCov input (predMu, predCov) newMeas =
  (newMu, newCov)
  where
    newMu  = predMu + kkMat #> (newMeas - upMu)
    newCov = sym $ (unSym predCov) - kkMat <> skMat <> tr kkMat

    predMu' = measure input predMu

    kkMat = ckMat <> unsafeInv skMat
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

    sqCov   = chol predCov
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
  -> (R n, Sym n)      -- ^ Current estimate
  -> R m               -- ^ New measurement
  -> (R n, Sym n)      -- ^ New (filtered) estimate
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
  -> (R n, Sym n)       -- ^ Future smoothed estimate
  -> (R n, Sym n)       -- ^ Present filtered estimate
  -> (R n, Sym n)       -- ^ Present smoothed estimate

runEKS sysEvol linEvol sysCov input (futMu, futCov') (curMu, curCov') =
  (smMu, smCov)
  where
    smMu  = curMu + gkMat #> (futMu - predMu)
    smCov = sym $ curCov + gkMat <> (futCov - predCov) <> tr gkMat

    futCov = unSym futCov'

    curCov = unSym curCov'

    (predMu, predCov') = runEKFPrediction sysEvol linEvol sysCov input (curMu, curCov')
    predCov  = unSym predCov'

    gkMat = curCov <> tr lin <> unsafeInv predCov
    lin   = linEvol input curMu


runKS
  :: (KnownNat n)
  => (a -> R n -> Sq n) -- ^ Linear system evolution at a point
  -> (a -> Sym n)       -- ^ Covariance matrix encoding system evolution noise
  -> a                  -- ^ Dynamical input
  -> (R n, Sym n)       -- ^ Future smoothed estimate
  -> (R n, Sym n)       -- ^ Present filtered estimate
  -> (R n, Sym n)       -- ^ Present smoothed estimate
runKS linEvol =
  runEKS (\inp sys -> linEvol inp sys #> sys) linEvol

-- | Unscented Kalman smoother
runUKS
  :: (KnownNat n)
  => (a -> R n -> R n) -- ^ System evolution function
  -> (a -> Sym n)      -- ^ Covariance matrix encoding system evolution noise
  -> a                 -- ^ Dynamical input
  -> (R n, Sym n)      -- ^ Future smoothed estimate
  -> (R n, Sym n)      -- ^ Present filtered estimate
  -> (R n, Sym n)      -- ^ Present smoothed estimate
runUKS evolve sysCov input (futMu, futCov') (curMu, curCov') =
  (smMu, smCov)
  where
    smMu  = curMu + gkMat #> (futMu - predMu)
    smCov = sym $ curCov + gkMat <> (futCov - predCov) <> tr gkMat

    futCov = unSym futCov'

    curCov = unSym curCov'

    (predMu, predCov') = runUKFPrediction evolve sysCov input (curMu, curCov')
    predCov  = unSym predCov'

    gkMat = dkMat <> unsafeInv predCov

    dkMat = sum $
            zipWith (\pres fut ->
                       (col $ weightCM * (pres - curMu)) <> (row $ fut - predMu)
                    )
            sigmaPoints sigmaPoints'

    sqCov   = chol predCov'
    sqRows = map (* sqrt 3) $ toRows sqCov
    sigmaPoints = map (curMu +) sqRows ++ map (curMu -) sqRows -- 2n points
    sigmaPoints' = map (evolve input) sigmaPoints

    -- hand tuned weights, more exlanation required
    weightCM = 1 / 6

unsafeInv :: KnownNat n => Sq n -> Sq n
unsafeInv m = fromJust $ linSolve m eye
