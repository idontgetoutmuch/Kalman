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

-- | Take the
--
--  * Prior mean @muPrior@,
--
--  * Prior variance @sigmaPrior@,
--
--  * Observation map (represented as a matrix) @bigH@,
--
--  * Observation noise @bigSigmaY@,
--
--  * State update function @littleA@,
--
--  * A function which return the Jacobian of the state update
--  function at a given point @bigABuilder@,
--
-- * State noise @bigSigmaX@,
--
-- * List of observations @ys@
--
-- and return the posterior mean and variance.
extKalman ::  forall m n .
              (KnownNat m, KnownNat n, (1 <=? n) ~ 'True, (1 <=? m) ~ 'True) =>
              R n -> Sq n ->
              L m n -> Sq m ->
              (R n -> R n) -> (R n -> Sq n) -> Sq n ->
              [R m] ->
              [(R n, Sq n)]
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
