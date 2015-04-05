{-# OPTIONS_GHC -Wall                     #-}
{-# OPTIONS_GHC -fno-warn-name-shadowing  #-}
{-# OPTIONS_GHC -fno-warn-type-defaults   #-}
{-# OPTIONS_GHC -fno-warn-unused-do-bind  #-}
{-# OPTIONS_GHC -fno-warn-missing-methods #-}
{-# OPTIONS_GHC -fno-warn-orphans         #-}

{-# LANGUAGE DataKinds                    #-}
{-# LANGUAGE ScopedTypeVariables          #-}
{-# LANGUAGE TypeOperators                #-}
{-# LANGUAGE TypeFamilies                 #-}

module Kalman (
  extKalman
  )where

import GHC.TypeLits
import Numeric.LinearAlgebra.Static hiding ( create )
import Data.Maybe ( fromJust )

extKalman ::  forall m n .
              (KnownNat m, KnownNat n, (1 <=? n) ~ 'True, (1 <=? m) ~ 'True) =>
              R n              -- ^ Prior mean
              -> Sq n          -- ^ Prior variance
              -> L m n         -- ^ Observation map (represented as a matrix)
              -> Sq m          -- ^ Observation noise
              -> (R n -> R n)  -- ^ State update function
              -> (R n -> Sq n) -- ^ A function which returns the
                               -- Jacobian of the state update
                               -- function at a given point
              -> Sq n          -- ^ State noise
              -> [R m]         -- ^ List of observations
              -> [(R n, Sq n)] -- ^ List of posterior means and variances
extKalman muPrior sigmaPrior bigH bigSigmaY
  littleA bigABuilder bigSigmaX ys = result
  where
    result = scanl update (muPrior, sigmaPrior) ys

    update :: (R n, Sq n) -> R m -> (R n, Sq n)
    update (xHatFlat, bigSigmaHatFlat) y =
      (xHatFlatNew, bigSigmaHatFlatNew)
      where

        v :: R m
        v = y - (bigH #> xHatFlat)

        bigS :: Sq m
        bigS = bigH <> bigSigmaHatFlat <> (tr bigH) + bigSigmaY

        bigK :: L n m
        bigK = bigSigmaHatFlat <> (tr bigH) <> (inv bigS)

        xHat :: R n
        xHat = xHatFlat + bigK #> v

        bigSigmaHat :: Sq n
        bigSigmaHat = bigSigmaHatFlat - bigK <> bigS <> (tr bigK)

        bigA :: Sq n
        bigA = bigABuilder xHat

        xHatFlatNew :: R n
        xHatFlatNew = littleA xHat

        bigSigmaHatFlatNew :: Sq n
        bigSigmaHatFlatNew = bigA <> bigSigmaHat <> (tr bigA) + bigSigmaX

inv :: (KnownNat n, (1 <=? n) ~ 'True) => Sq n -> Sq n
inv m = fromJust $ linSolve m eye
