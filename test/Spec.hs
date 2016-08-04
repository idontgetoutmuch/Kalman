{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TupleSections #-}

module Main where

import Data.Random
import Data.Random.Distribution.MultiNormal
import Numeric.Kalman

import Control.Monad (replicateM)
--import GHC.TypeLits

import Graphics.Rendering.Chart.Easy          
import Graphics.Rendering.Chart.Backend.Cairo

import Numeric.LinearAlgebra.Static hiding (mean)

--import Control.Monad (replicateM)
--import Data.Maybe (fromJust)

import qualified Data.List as L
import qualified Numeric.LinearAlgebra as LA

main :: IO ()
--main = putStrLn "Test suite not yet implemented"

--main = runUKFTest1
main = runNoisySine1

---------------------------------------
--- Unscented Kalman filter testing ---
---------------------------------------
runUKFTest1 :: IO ()
runUKFTest1 = do
  -- polar coordinate transform test
  let numSamples = 500
      dt = 0.01
      tVals = take (numSamples + 1) [0,dt..] :: [Double] -- +1 because t = 0 is our initial estimate
  
      initSys = MultiNormal (vector [0, 1]) (sym . diag . vector $ [4e-4, 2.7e-5] :: Sym 2)
      initEst = MultiNormal (vector [1, pi/2]) (sym . diag . vector $ [0.02^2, (15*pi/180)^2] :: Sym 2)

  let coordTrans :: R 2 -> R 2
      coordTrans polar =
        let [r,t] = LA.toList $ extract polar
        in vector [r * cos t, r * sin t]

  measurements <- map coordTrans <$> replicateM numSamples (sample initEst)

  let samples = map (\meas ->
                       let [x,y] = LA.toList $ extract meas
                       in (x,y)
                    ) measurements

      ukf   = scanl (runUKF (const coordTrans)
                            (const (sym . diag . vector $ [0.02^2, (15*pi/180)^2] :: Sym 2))
                            (const id)
                            (const (sym . diag . vector $ [0,0])) 1)
                    initSys measurements

      ukpts = map (\kf ->
                       let [x,y] = LA.toList . extract $ mu kf
                       in (x,y)
                    ) ukf

  let linMeas :: R 2 -> L 2 2
      linMeas est =
        let [x, _] = LA.toList $ extract est
        in matrix [1,0, 0,x]

  let ekf = scanl (runEKF (const coordTrans)
                          (const linMeas)
                          (const (sym . diag . vector $ [0.02^2, (15*pi/180)^2] :: Sym 2))
                          (const id)
                          (const (const eye))
                          (const (sym . diag . vector $ [0,0])) 1)
                  initSys measurements

      ekpts = map (\kf ->
                       let [x,y] = LA.toList . extract $ mu kf
                       in (x,y)
                    ) ekf

      -- predTest = runUKFPrediction id (sym . diag . vector $ [0,0]) initSys
      -- updtTest = runUKFUpdate coordTrans (sym . diag . vector $ [0.02^2, (15*pi/180)^2] :: Sym 2) initSys (head measurements)

  -- print $ extract . mu $ predTest
  -- print $ extract . unSym . cov $ predTest
  -- putStrLn "\n"
  -- print $ extract . mu $ updtTest
  -- print $ extract . unSym . cov $ updtTest
  -- putStrLn "\n"
  -- print $ last ukpts
  -- print $ last ekpts

  toFile def "noisy-coords.png" $ do
    layout_title .= "Noisy coord transform"
    setColors [opaque black, opaque blue
              ,opaque red, opaque green
              --,opaque hotpink
              ,opaque lightgrey, opaque lightgrey
              ,opaque lightslategrey, opaque lightslategrey
              ]
    plot (points "True signal"  [(0,1)])
    plot (points "Measurements" samples)
    plot (points "UKF estimates" ukpts)
    plot (points "EKF estimates" ekpts)

  toFile def "noisy-params.png" $ do
    layout_title .= "Noisy coord transform"
    setColors [opaque black, opaque black
              ,opaque green, opaque green
              ,opaque red, opaque red
              --,opaque hotpink
              ,opaque lightgrey, opaque lightgrey
              ,opaque lightslategrey, opaque lightslategrey
              ]
    plot (line "True X signal"  [[(head tVals, 0), (last tVals, 0)]])
    plot (line "True Y signal"  [[(head tVals, 1), (last tVals, 1)]])
    plot (points "UKF X estimates" $ zipWith (\t (x,_) -> (t, x)) tVals ukpts)
    plot (points "UKF Y estimates" $ zipWith (\t (_,y) -> (t, y)) tVals ukpts)
    plot (points "EKF X estimates" $ zipWith (\t (x,_) -> (t, x)) tVals ekpts)
    plot (points "EKF Y estimates" $ zipWith (\t (_,y) -> (t, y)) tVals ekpts)

---------------------
--- Noisy sine # 1---
---------------------
-- In this example, we estimate the parameters (amplitude, period, vertical shift, horizontal shift)
-- defining sinusoidal evolution.
-- Figure 3 in my notes

runNoisySine1 :: IO ()
runNoisySine1 = do

  let (a, p, v) = (0.7, 2.2*pi, -0.5) -- (hidden) internal system parameters
      (a0, p0, v0) = (0.8, 2*pi, -0.6)    -- initial estimates
      --(aSig, pSig, vSig) = (0.2, 0.2*pi, 0.001) -- initial uncertainties
      measCov = 0.01

      --- Figure 1 in my notes
      --numSamples = 499
      --dt = 0.002
      --- Figure 2 in my notes
      numSamples = 199
      dt = 0.02
      tVals = take (numSamples + 1) [0,dt..] -- +1 because t = 0 is our initial estimate
  
      --initSys = vector [a, p, v] -- initial (true) system: (parameter1, parameter2, state)
      --initEst = MultiNormal (vector [a0, p0, v0, a0*p0, v0]) (sym . diag . vector $ [0.04, 0.4, 0.04, 0.001, 0.001] :: Sym 5)
      initEst = MultiNormal (vector [a0, p0, v0, a0*p0, v0])
                (sym . matrix $ [0.04,  0,    0,    1e-5, -2e-3,
                                 0,     0.4,  0,    1e-2, 12e-3,
                                 0,     0,    0.04, 1e-4, 1e-3,
                                 1e-5,  1e-2, 1e-4, 1e-3, 1e-4,
                                 -2e-3, 12e-3, 1e-3, 1e-4, 1e-3] :: Sym 5)

      evolMat :: R 5 -> Sq 5
      evolMat cur =
        let pCur  = (extract cur) LA.! 1
            p2dt  = pCur * pCur * dt
        in matrix [1,0,0,0,0, 0,1,0,0,0, 0,0,1,0,0, 0,0,p2dt,1,-p2dt, 0,0,0,dt,1]

      evolSys :: Double -> R 5 -> R 5
      evolSys t cur =
        let [a,p,v,ap,b] = LA.toList $ extract cur
        in vector [a, p, v, a * p * cos (p * t), a * sin (p * t) + v]
  
      sysCovariance = sym . diag . vector $ [0,0,0,5e-4,7e-5] :: Sym 5

      measMat = row (vector [0,0,0,0,1]) :: L 1 5
      measTrans = (measMat #>)
      measCovariance = sym . diag . vector $ [measCov] :: Sym 1
      measurementNoise = MultiNormal (vector [0]) measCovariance
      
      trueSystem = [vector [a, p, v, a * p * cos(p*t), a * sin(p*t) + v] | t <- tVals] :: [R 5]

  --trueSystem   <- simulateSystem numSamples sysModel initSys
  measurements <- mapM (\sys -> do
                            noise <- sample measurementNoise
                            return $ measTrans sys + noise
                       ) $ tail trueSystem

  --mapM_ print $ map (LA.toList . extract) trueSystem
  -- print $ map ((LA.! 0) . extract) measurements

  let truesine = zip tVals $ map ((LA.! 4) . extract) trueSystem
      samples  = zip (tail tVals) $ map ((LA.! 0) . extract) measurements

      kfiltered = scanl (runKF (const (const measMat)) (const measCovariance) (const evolMat) (const sysCovariance) 1) initEst measurements
      ekf = scanl (\sys (t,meas) ->
                     runEKF (\_t sys -> measMat #> sys)
                     (\_t _sys -> measMat)
                     (const measCovariance)
                     evolSys
                     (const evolMat)
                     (const sysCovariance)
                     t sys meas
                  ) initEst $ zip (tail tVals) measurements

      ukf = scanl (\sys (t,meas) ->
                     runUKF (\_t sys -> measMat #> sys)
                     (const measCovariance)
                     evolSys
                     (const sysCovariance)
                     t sys meas
                  ) initEst $ zip (tail tVals) measurements

      kfstates  = zipWith (\t kf -> (t, (extract . mu $ kf) LA.! 4)) tVals kfiltered
      ekfstates = zipWith (\t kf -> (t, (extract . mu $ kf) LA.! 4)) tVals ekf
      ukfstates = zipWith (\t kf -> (t, (extract . mu $ kf) LA.! 4)) tVals ukf
  
      [kfa,kfp,kfv] = L.transpose $ zipWith (\t kf -> map (t,) . take 3 . LA.toList . extract . mu $ kf) tVals kfiltered
      [ekfa,ekfp,ekfv] = L.transpose $ zipWith (\t kf -> map (t,) . take 3 . LA.toList . extract . mu $ kf) tVals ekf
      [ukfa,ukfp,ukfv] = L.transpose $ zipWith (\t kf -> map (t,) . take 3 . LA.toList . extract . mu $ kf) tVals ukf

      [kfaCovs, kfpCovs, kfvCovs, kfsCovs] = L.transpose $
                                             map (\kf -> let kfcov = extract . unSym $ cov kf
                                                         in [kfcov LA.! 0 LA.! 0, kfcov LA.! 1 LA.! 1,
                                                             kfcov LA.! 2 LA.! 2, kfcov LA.! 4 LA.! 4])
                                             kfiltered

      kfsUpper1 = zipWith (\(t, kfs) kfsCov -> (t, kfs + sqrt kfsCov)) kfstates kfsCovs
      kfsLower1 = zipWith (\(t, kfs) kfsCov -> (t, kfs - sqrt kfsCov)) kfstates kfsCovs

      kfsUpper2 = zipWith (\(t, kfs) kfsCov -> (t, kfs + 2 * sqrt kfsCov)) kfstates kfsCovs
      kfsLower2 = zipWith (\(t, kfs) kfsCov -> (t, kfs - 2 * sqrt kfsCov)) kfstates kfsCovs

      kfaUpper1 = zipWith (\(t, kfs) kfsCov -> (t, kfs + sqrt kfsCov)) kfa kfaCovs
      kfaLower1 = zipWith (\(t, kfs) kfsCov -> (t, kfs - sqrt kfsCov)) kfa kfaCovs

      kfaUpper2 = zipWith (\(t, kfs) kfsCov -> (t, kfs + 2 * sqrt kfsCov)) kfa kfaCovs
      kfaLower2 = zipWith (\(t, kfs) kfsCov -> (t, kfs - 2 * sqrt kfsCov)) kfa kfaCovs

      kfpUpper1 = zipWith (\(t, kfs) kfsCov -> (t, kfs + sqrt kfsCov)) kfp kfpCovs
      kfpLower1 = zipWith (\(t, kfs) kfsCov -> (t, kfs - sqrt kfsCov)) kfp kfpCovs

      kfpUpper2 = zipWith (\(t, kfs) kfsCov -> (t, kfs + 2 * sqrt kfsCov)) kfp kfpCovs
      kfpLower2 = zipWith (\(t, kfs) kfsCov -> (t, kfs - 2 * sqrt kfsCov)) kfp kfpCovs

      kfvUpper1 = zipWith (\(t, kfs) kfsCov -> (t, kfs + sqrt kfsCov)) kfv kfvCovs
      kfvLower1 = zipWith (\(t, kfs) kfsCov -> (t, kfs - sqrt kfsCov)) kfv kfvCovs

      kfvUpper2 = zipWith (\(t, kfs) kfsCov -> (t, kfs + 2 * sqrt kfsCov)) kfv kfvCovs
      kfvLower2 = zipWith (\(t, kfs) kfsCov -> (t, kfs - 2 * sqrt kfsCov)) kfv kfvCovs

      ksmoothed = scanr1 (\pres fut -> runKS (const evolMat) (const sysCovariance) 1 fut pres) kfiltered
      ksstates  = zipWith (\t ks -> (t, (extract . mu $ ks) LA.! 4)) tVals ksmoothed

  --mapM_ print $ map (LA.toList . extract . mu) kfiltered
  --mapM_ print kfa
  
  toFile def "noisy-sine-points.png" $ do
    layout_title .= "Noisy sine (state estimate)"
    setColors [opaque black, opaque blue
              ,opaque red, opaque purple, opaque hotpink, opaque green
              ,opaque lightgrey, opaque lightgrey
              ,opaque lightslategrey, opaque lightslategrey]
    plot (line   "True signal"  [truesine])
    plot (points "Measurements" samples)
    plot (line   "Recursively estimated signal" [kfstates])
    plot (line   "Recursively estimated signal" [ekfstates])
    plot (line   "Recursively estimated signal" [ukfstates])
    plot (line   "Recursively smoothed signal"  [ksstates])
    plot (line   "Filtered 1 standard deviation" [kfsUpper1])
    plot (line   "Filtered 1 standard deviation" [kfsLower1])
    plot (line   "Filtered 2 standard deviations" [kfsUpper2])
    plot (line   "Filtered 2 standard deviations" [kfsLower2])


  let truea = [(head tVals, a), (last tVals, a)]
      truep = [(head tVals, p), (last tVals, p)]
      truev = [(head tVals, v), (last tVals, v)]

  toFile def "noisy-sine-amplitude.png" $ do
    layout_title .= "Noisy sine (amplitude estimate)"
    setColors [opaque black, opaque red, opaque purple, opaque hotpink
              ,opaque lightgrey, opaque lightgrey
              ,opaque lightslategrey, opaque lightslategrey
              ]
    plot (line   "True signal"  [truea])
    plot (line   "Recursively estimated signal" [kfa])
    plot (line   "Recursively estimated signal" [ekfa])
    plot (line   "Recursively estimated signal" [ukfa])
    plot (line   "Filtered 1 standard deviation" [kfaUpper1])
    plot (line   "Filtered 1 standard deviation" [kfaLower1])
    plot (line   "Filtered 2 standard deviations" [kfaUpper2])
    plot (line   "Filtered 2 standard deviations" [kfaLower2])

  toFile def "noisy-sine-period.png" $ do
    layout_title .= "Noisy sine (period estimate)"
    setColors [opaque black, opaque red, opaque purple, opaque hotpink
              ,opaque lightgrey, opaque lightgrey
              ,opaque lightslategrey, opaque lightslategrey
              ]
    plot (line   "True signal"  [truep])
    plot (line   "Recursively estimated signal" [kfp])
    plot (line   "Recursively estimated signal" [ekfp])
    plot (line   "Recursively estimated signal" [ukfp])
    plot (line   "Filtered 1 standard deviation" [kfpUpper1])
    plot (line   "Filtered 1 standard deviation" [kfpLower1])
    plot (line   "Filtered 2 standard deviations" [kfpUpper2])
    plot (line   "Filtered 2 standard deviations" [kfpLower2])

  toFile def "noisy-sine-vertical.png" $ do
    layout_title .= "Noisy sine (vertical estimate)"
    setColors [opaque black, opaque red, opaque purple, opaque hotpink
              ,opaque lightgrey, opaque lightgrey
              ,opaque lightslategrey, opaque lightslategrey
              ]
    plot (line   "True signal"  [truev])
    plot (line   "Recursively estimated signal" [kfv])
    plot (line   "Recursively estimated signal" [ekfv])
    plot (line   "Recursively estimated signal" [ukfv])
    plot (line   "Filtered 1 standard deviation" [kfvUpper1])
    plot (line   "Filtered 1 standard deviation" [kfvLower1])
    plot (line   "Filtered 2 standard deviations" [kfvUpper2])
    plot (line   "Filtered 2 standard deviations" [kfvLower2])

  putStrLn $ "Kalman processing complete."

